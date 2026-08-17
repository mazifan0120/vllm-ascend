# DualPath 下线 abort 专用设计（close 协议 + Reverse destination hold）实施计划

> **历史计划（于 2026-08-17 被取代）：** 本文保留
> `CloseReverseAttempt` / `HoldLedger` 下线过程的历史记录；其中仍保留的
> elapsed-time watchdog、Decision-only channel 和 two-field request key 描述，
> 已由 [DualPath ABORT Notification and Watchdog Removal](../specs/2026-08-16-dual-path-abort-notification-and-watchdog-removal.md)
> 取代，不再是当前实现契约。

<!-- End of the 2026-08-17 supersession notice. -->

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除 DualPath 中为"PE 侧请求在 Reverse 完成前异常终止"而设计的整套 CloseReverseAttempt 协议和 Reverse destination hold，让 abort 回到 main 的语义（立即释放、不等证明），同时完整保留 `reverse_attempt_id` epoch 机制。

**Architecture:** abort 专用设计分三层：(A) close 协议全链路，入口是 `_delay_free_for_connector` 里的 `FINISHED_ABORTED` 分支；(B) `HoldLedger` 与 Reverse destination hold，表面通用但正常路径下是纯冗余，只在请求提前终止时起作用；(C) 看着像 abort 实则是 epoch 机制的部分（`_register_decision_locked` 单调性、`_closed_through_attempt_ids` 水位、I4 gate），必须原样保留。按 A → B → 连带修复的顺序删除，每个 Task 结束时全量单测必须绿。

**Tech Stack:** Python 3.12、pytest、msgspec、pyzmq、vLLM v1 KVConnector 接口、ruff 0.14.0

## Global Constraints

- **必须保留的 epoch 机制**（任何 Task 都不得触碰）：`PathDecisionResult.reverse_attempt_id`、`reverse_attempt_id = num_preemptions` 的赋值（`path_decision.py:270`）、`_register_decision_locked` 的去重与单调性、`_close_skipped_lower_attempts_locked`、`DecisionReplyStatus.STALE_CLOSED`、`_closed_through_attempt_ids` 在 `_register_decision_locked` 路径上的读写、`_activate_received_decision` 里的 `result.reverse_attempt_id > latest_attempt` 判断、`_waiting_reverse_attempt_ids` 与 I4 gate（`_close_reverse_completion_job`）。
- **必须保留的 job 机制**：`JobLedger`、`JobKind.REVERSE_COMPLETION`（PE 放行请求的唯一证据）、`JobKind.REVERSE_SEND`（DE 侧延迟释放源块的依据）。
- **wire format**：`encode_control_message` / `decode_control_message` 的 envelope 结构（`{"kind": ..., "payload": ...}`）保持不变，只删 `ControlMessageKind` 的成员。不要把 envelope 拍平。
- **跑测试必须在非沙箱环境**：沙箱下 `test_decode_admission_integration.py` 会因 `sysctl` 权限报 `CalledProcessError`，是环境噪声。执行 shell 时带 `required_permissions: ["all"]`。
- **lint 命令**：`uvx ruff@0.14.0 check <paths>` 和 `uvx ruff@0.14.0 format --check <paths>`（venv 内没有装 ruff，也没有 pip）。
- **commit 必须签名**：`git commit -s`，遵循 Conventional Commits，type 用 `refactor`。
- 全量回归命令（每个 Task 的最后一步都要跑）：
  ```bash
  .venv/bin/python -m pytest tests/ut/distributed/kv_transfer/ -q --no-header
  ```
  基线：**694 passed**（本计划开始时的状态）。每个 Task 之后这个数字会下降，计划里给出了每步的预期值。

---

## 文件职责总览

| 文件 | 本计划中的角色 |
| --- | --- |
| `vllm_ascend/.../dual_path/close_registry.py` | 整个删除（100 行，纯 close 判定表） |
| `vllm_ascend/.../dual_path/scheduler.py` | 删 close 驱动、abort 分支、hold 获取与预算 |
| `vllm_ascend/.../dual_path/path_decision_channel.py` | 删 close 消息类型、close 注册表、四个 mark/claim 方法 |
| `vllm_ascend/.../dual_path/ledgers.py` | 删 `HoldLedger`/`HoldKind`/`HoldRecord`，只留 `JobLedger` |
| `vllm_ascend/envs.py` | 删 3 个环境变量 |
| `tests/.../dual_path/test_close_reverse_attempt.py` | 整个删除（404 行） |
| `tests/.../dual_path/test_hold_ledger.py` | 改名为 `test_job_ledger.py`，只留 JobLedger 部分 |
| `tests/.../dual_path/test_dual_path_connector.py` | 契约清单，是本计划所有"失败的测试"的主要载体 |
| `tests/.../dual_path/test_split_race_cleanup.py` | 第二份方法名清单，需同步 |
| `tests/.../dual_path/test_watchdogs_limits.py` | 删 hold 压力测试，改 watchdog 断言 |
| `tests/.../dual_path/test_audit_fixes.py` | 删 close 相关回归 |
| `tests/.../dual_path/test_channel_registry.py` | 改 1 个信封测试，其余（epoch 矩阵）保留 |
| `tests/.../dual_path/test_de_reverse_send_proof.py` | 改断言，从 `_is_reverse_send_complete` 换成直接读 job ledger |
| `tests/.../dual_path/test_forward_abort_release.py` | 1 处 hold 断言改写 |
| `tests/.../dual_path/test_topology_validation.py` | 2 处 hold 断言删除 |
| `tests/.../dual_path/test_resume_admission.py` | 2 组"不新增记录"断言从 hold 换成 job |
| `tests/.../dual_path/test_i4_gate.py` | **不动**（纯 epoch/I4） |
| `tests/.../dual_path/test_reverse_attempt_identity.py` | **不动**（纯 epoch） |

**路径前缀**：源码全部在 `vllm_ascend/distributed/kv_transfer/kv_p2p/`，测试全部在 `tests/ut/distributed/kv_transfer/`。下文为可读性省略前缀。

---

## Phase A — close 协议下线

### Task 1: 删除 PE 侧 close 驱动与 abort 分支

删掉 close 机制的**触发端**。做完这一步后 `path_decision_channel.py` 里的 `submit_close` / `_handle_close_frame` 等仍然存在但无人调用，代码可以正常运行——这是有意的，把通道层留到 Task 2 单独删，出问题好定位。

**Files:**

- Modify: `dual_path/scheduler.py`
- Modify: `tests/dual_path/test_dual_path_connector.py`（字段清单 ~L676-706、方法清单 ~L1554-1570）
- Modify: `tests/dual_path/test_split_race_cleanup.py`（方法清单 ~L588-603）
- Modify: `tests/dual_path/test_watchdogs_limits.py`
- Modify: `tests/dual_path/test_audit_fixes.py`
- Delete: `tests/dual_path/test_close_reverse_attempt.py`

**Interfaces:**

- Consumes: 无（第一个 Task）
- Produces: `DualPathConnectorScheduler` 不再有 `_pending_close_futures` / `_pending_close_requests` / `_close_retry_deadlines` / `_pending_ordinary_release` 四个字段，不再有 `_initiate_reverse_attempt_close` / `_submit_reverse_attempt_close` / `_reconcile_reverse_attempt_closes` 三个方法。`_delay_free_for_connector(request) -> bool` 签名不变，但 prefill 分支恒返回 `False`。

---

- [ ] **Step 1: 从契约清单里删掉四个字段（这就是"失败的测试"）**

编辑 `tests/dual_path/test_dual_path_connector.py`，在 `dual_path_scheduler_fields` 集合中删除这四行：

```python
            "_pending_ordinary_release",
            "_pending_close_futures",
            "_pending_close_requests",
            "_close_retry_deadlines",
```

注意 `_pending_ordinary_release` 和另外三个不相邻（前者在 `_reverse_destination_holds` 之后，后三者在 `_latest_reverse_attempt_ids` 之后）。

- [ ] **Step 2: 从两份方法清单里删掉三个方法名**

`tests/dual_path/test_dual_path_connector.py` 和 `tests/dual_path/test_split_race_cleanup.py` 各有一份 `DualPathConnectorScheduler` 的方法名清单。两个文件里都删除这三行：

```python
        "_initiate_reverse_attempt_close",
        "_submit_reverse_attempt_close",
        "_reconcile_reverse_attempt_closes",
```

保留同一清单里的 `_sweep_prefill_recovery_watchdogs` 和 `_close_reverse_completion_job`——前者是超时保护（非 abort），后者是 I4 gate。

- [ ] **Step 3: 运行契约测试，确认失败**

```bash
.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py -q --no-header
```

预期：FAIL，报 `Items in the first set but not the second` 之类，列出 `_pending_close_futures` 等仍存在于实现中。

- [ ] **Step 4: 删除 `_delay_free_for_connector` 的 abort 分支**

`dual_path/scheduler.py`，`_delay_free_for_connector` 末尾（约 L1809-1818）整段删除，让 decode 分支之后直接 `return False`：

```python
        if (
            getattr(request, "status", None) is RequestStatus.FINISHED_ABORTED
            and request_id in self._waiting_reverse_attempt_ids
        ):
            # Abort while waiting for the Reverse: ordinary ownership is freed
            # through the finished_recving injection, but the Reverse
            # destination hold is retained until the close proves SAFE (I8).
            self._pending_ordinary_release.add(request_id)
            self._initiate_reverse_attempt_close(request_id)
        return False
```

替换为单独一行 `return False`。同时把方法开头的注释改成反映新语义：

```python
    def _delay_free_for_connector(self, request: Request) -> bool:
        # Neither direction delays the free on the Prefill side: Forward is the
        # ordinary Layerwise push, and an aborted Reverse destination follows
        # the parent's immediate-free semantics. Only a Decode request with an
        # open reverse-send job is held back, so the engine keeps stepping
        # until the send job reports.
```

删除后检查文件顶部的 `RequestStatus` import 是否还有其他使用点；如果没有，一并删掉该 import。

- [ ] **Step 5: 删除三个 close 驱动方法**

`dual_path/scheduler.py`，删除 `_initiate_reverse_attempt_close`（约 L525-536）、`_submit_reverse_attempt_close`（约 L538-549）、`_reconcile_reverse_attempt_closes`（约 L551-578）三个方法的完整定义。

- [ ] **Step 6: 删除 `update_connector_output` 里的 close 调用与注入**

`dual_path/scheduler.py`，`update_connector_output` 开头删除这一行：

```python
        self._reconcile_reverse_attempt_closes()
```

同方法末尾删除整个 `_pending_ordinary_release` 注入块：

```python
        if self._pending_ordinary_release:
            if connector_output.finished_recving is None:
                connector_output.finished_recving = set()
            connector_output.finished_recving.update(self._pending_ordinary_release)
            self._pending_ordinary_release.clear()
```

- [ ] **Step 7: 删除 watchdog 清扫里的 close 清理**

`dual_path/scheduler.py`，`_sweep_prefill_recovery_watchdogs` 循环体中删除这三行（约 L1621-1623）：

```python
            self._close_retry_deadlines.pop(request_id, None)
            self._pending_close_futures.pop(request_id, None)
            self._pending_close_requests.pop(request_id, None)
```

紧邻其上的 `self._recovery_deadlines.pop(request_id)` 保留，紧邻其下的 `self._prefill_invalid_request_ids.add(request_id)` 及之后的控制失败逻辑也全部保留——超时保护本身不属于 abort 设计。

- [ ] **Step 8: 删除四个字段与相关 import**

`dual_path/scheduler.py` 的 `__init__` 中删除：

```python
        self._pending_ordinary_release: set[str] = set()
        self._pending_close_futures: dict[str, Future[CloseReplyStatus]] = {}
        self._pending_close_requests: dict[str, tuple[CloseReverseAttempt, DecodeControlEndpoint]] = {}
        self._close_retry_deadlines: dict[str, float] = {}
```

同时删除 `self._close_retry_backoff_s` 赋值行，以及 `__init__` 里那个环境变量校验元组中的这一行：

```python
            ("VLLM_ASCEND_DUALPATH_CLOSE_RETRY_BACKOFF_S", self._close_retry_backoff_s),
```

从 `path_decision_channel` 的 import 列表里删掉 `CloseReplyStatus` 和 `CloseReverseAttempt`。检查 `Future` 和 `DecodeControlEndpoint` 是否还有其他使用点（`DecodeControlEndpoint` 仍被 `DualPathDecisionMetadata` 用到，`Future` 仍被 `_prefill_delivery_futures` 用到，两者都应保留）。

- [ ] **Step 9: 删除环境变量 `VLLM_ASCEND_DUALPATH_CLOSE_RETRY_BACKOFF_S`**

`vllm_ascend/envs.py` 中删除这一项及其注释：

```python
    # Backoff in integer seconds between identical CloseReverseAttempt retries
    # after a NOT_SAFE reply. Default 1; valid range is greater than zero; not
    # sensitive. Tunable per spec section 11.
    "VLLM_ASCEND_DUALPATH_CLOSE_RETRY_BACKOFF_S": lambda: int(
        os.getenv("VLLM_ASCEND_DUALPATH_CLOSE_RETRY_BACKOFF_S", "1")
    ),
```

同时把 `VLLM_ASCEND_DUALPATH_RECOVERY_WATCHDOG_S` 的注释里提到 close 的部分改掉，新注释：

```python
    # DualPath PE Reverse-completion watchdog in integer seconds: bounds how
    # long a parked DE_READ request may wait for its Reverse completion before
    # the request is failed. Default 30; valid range is greater than zero; not
    # sensitive. Tunable per spec section 11 — the default requires NPU
    # workload measurement.
```

- [ ] **Step 10: 删除 `test_close_reverse_attempt.py`**

```bash
rm tests/ut/distributed/kv_transfer/dual_path/test_close_reverse_attempt.py
```

该文件 404 行全部是 close 矩阵、丢包重试幂等、abort 双释放的测试，无一可留。

- [ ] **Step 11: 清理 `test_watchdogs_limits.py` 中的 close 用例**

删除 `TestCleanupRetention` 类中的 `test_late_close_for_removed_admission_safe_with_proof_not_safe_otherwise` 整个方法。

修改 `test_env_vars_registered_with_defaults`，从被断言的环境变量清单里去掉 `VLLM_ASCEND_DUALPATH_CLOSE_RETRY_BACKOFF_S`。

`TestWatchdogs::test_pe_recovery_watchdog_expiry_fails_request_without_releasing_holds` 本 Task 暂不改（hold 还在），留到 Task 3。

- [ ] **Step 12: 清理 `test_audit_fixes.py` 中依赖 close 驱动的用例**

用下面的命令定位所有需要处理的位置：

```bash
rg -n "close|Close" tests/ut/distributed/kv_transfer/dual_path/test_audit_fixes.py
```

删除其中调用 `scheduler._initiate_reverse_attempt_close`、`scheduler._reconcile_reverse_attempt_closes`、`_pending_close_futures`、`_pending_ordinary_release` 的测试方法。仅调用 `receiver.claim_reverse_activation` / `mark_reverse_work_published` / `mark_reverse_send_complete`（通道层）的用例本 Task 保留，Task 2 再处理。

- [ ] **Step 13: 运行全量测试**

```bash
.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/ -q --no-header
```

预期：全绿。总数从 694 降到约 660（删掉 `test_close_reverse_attempt.py` 的 ~28 个用例和其他零散用例）。如果有失败，逐个看是否属于"测试还在断言已删除的行为"，是则修测试，不是则说明删多了。

- [ ] **Step 14: lint**

```bash
uvx ruff@0.14.0 check vllm_ascend/distributed/kv_transfer vllm_ascend/envs.py tests/ut/distributed/kv_transfer
uvx ruff@0.14.0 format --check vllm_ascend/distributed/kv_transfer vllm_ascend/envs.py tests/ut/distributed/kv_transfer
```

有问题用 `uvx ruff@0.14.0 check --fix` 和 `uvx ruff@0.14.0 format` 修。

- [ ] **Step 15: Commit**

```bash
git add -A vllm_ascend/distributed/kv_transfer vllm_ascend/envs.py tests/ut/distributed/kv_transfer
git commit -s -m "refactor(dual_path): drop the PE-side CloseReverseAttempt driver

An aborted DE_READ request now follows the parent's immediate-free
semantics instead of retaining its Reverse destination until a remote
SAFE proof arrives. Removes the abort branch in _delay_free_for_connector
and the three close-driving methods it fed."
```

---

### Task 2: 删除控制通道 close 协议与 `close_registry` 模块

删掉 close 的**服务端**。做完这一步，控制通道只剩 Decision 一种消息。

**Files:**

- Delete: `dual_path/close_registry.py`
- Modify: `dual_path/path_decision_channel.py`
- Modify: `dual_path/scheduler.py`（删四个 coordinator 方法的调用点）
- Modify: `tests/dual_path/test_channel_registry.py`
- Modify: `tests/dual_path/test_audit_fixes.py`
- Modify: `tests/dual_path/test_path_decision_channel.py`
- Modify: `tests/dual_path/test_dual_path_connector.py`（若含 coordinator 方法清单）

**Interfaces:**

- Consumes: Task 1 的产物（scheduler 已无 close 驱动）
- Produces: `PathDecisionCoordinator` 不再有 `submit_close` / `claim_reverse_activation` / `cancel_reverse_publication` / `mark_reverse_work_published` / `mark_reverse_send_complete` / `_handle_close_frame`，不再有 `_closed_reverse_records` / `_reverse_attempt_states` 两个注册表。`ControlMessageKind` 只剩 `DECISION` 一个成员。`_closed_through_attempt_ids` **保留**（epoch 水位）。

---

- [ ] **Step 1: 写失败的测试——信封只认 Decision**

编辑 `tests/dual_path/test_channel_registry.py` 的 `TestMessageKindEnvelope::test_message_kind_decodes_decision_vs_close`，替换为：

```python
class TestMessageKindEnvelope:
    def test_decision_envelope_round_trips_and_unknown_kind_is_rejected(self):
        decision = _decision(attempt_id=0)
        kind, payload = channel.decode_control_message(channel.encode_path_decision(decision))
        assert kind is channel.ControlMessageKind.DECISION
        assert PathDecision.from_dict(payload) == decision

        foreign = msgspec.msgpack.encode({"kind": "CloseReverseAttempt", "payload": {}})
        with pytest.raises(PathDecisionValidationError):
            channel.decode_control_message(foreign)

    def test_control_message_kind_has_only_the_decision_member(self):
        assert [member.value for member in channel.ControlMessageKind] == ["Decision"]
```

在文件顶部加 `import msgspec`。`pytest`、`PathDecision`、`PathDecisionValidationError` 已经是该文件现有的 import（`channel` 是 `path_decision_channel` 的模块别名，`PathDecision` 等则是直接导入的名字，不要写成 `channel.PathDecision`）。

- [ ] **Step 2: 运行，确认失败**

```bash
.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/test_channel_registry.py -q --no-header
```

预期：`test_control_message_kind_has_only_the_decision_member` FAIL，因为枚举还有两个成员；另一个用例的 `pytest.raises` 也会 FAIL，因为 `CloseReverseAttempt` 目前是合法 kind。

- [ ] **Step 3: 收缩 `ControlMessageKind` 并删除 close 消息类型**

`dual_path/path_decision_channel.py`：

```python
class ControlMessageKind(str, Enum):
    DECISION = "Decision"
```

删除 `CloseReplyStatus` 枚举、`CloseReverseAttempt` dataclass（约 L215-242）、`encode_close_reply` 和 `decode_close_reply` 两个函数。

把 `_encode_status` / `_decode_status` 的类型注解从联合类型收窄：

```python
def _encode_status(status: DecisionReplyStatus) -> bytes:
    return msgspec.msgpack.encode({"status": status.value})


def _decode_status(payload: bytes, enum_cls: type) -> DecisionReplyStatus:
```

收窄之后 `decode_decision_reply` 尾部的 `# type: ignore[return-value]` 不再需要，一并删掉。`encode_control_message` / `decode_control_message` 的 envelope 结构不动。

- [ ] **Step 4: 删除接收端 close 分发与处理**

`dual_path/path_decision_channel.py` 的 `_handle_frames` 中删除整个 close 分支，改为：

```python
        identity, _, payload = frames
        try:
            kind, message_payload = decode_control_message(payload)
        except PathDecisionValidationError:
            logger.warning("path decision result receiver rejected malformed payload")
            return
        if kind is not ControlMessageKind.DECISION:
            logger.warning("path decision result receiver rejected non-Decision control message")
            return
        self._handle_decision_frame(socket, identity, message_payload)
```

删除 `_handle_close_frame` 方法的完整定义（约 L623-653）。

- [ ] **Step 5: 删除投递端**

`dual_path/path_decision_channel.py` 中删除 `_deliver_close` 函数（约 L357-386）和 `PathDecisionCoordinator.submit_close` 方法（约 L528-548）。

- [ ] **Step 6: 删除 close 注册表与四个状态方法**

`dual_path/path_decision_channel.py` 的 `PathDecisionCoordinator.__init__` 中删除：

```python
        self._reverse_attempt_states: dict[ReverseAttemptKey, ReverseAttemptRegistryState] = {}
        self._closed_reverse_records: dict[ReverseAttemptKey, ClosedReverseAttemptRecord] = {}
```

**保留** `self._closed_through_attempt_ids`——它是 epoch 水位，`_register_decision_locked` 依赖它。

删除 `claim_reverse_activation`、`cancel_reverse_publication`、`mark_reverse_work_published`、`mark_reverse_send_complete` 四个方法的完整定义（约 L655-707）。

`close()` 方法中删除这两行：

```python
                self._reverse_attempt_states.clear()
                self._closed_reverse_records.clear()
```

`unregister()` 中把注释改成只描述 epoch 水位的清理：

```python
    def unregister(self, key: DualPathRequestKey) -> None:
        with self._registry_lock:
            self._pending_keys.discard(key)
            self._accepted_decisions.pop(key, None)
            self._closed_through_attempt_ids.pop(key, None)
```

删除文件顶部对 `close_registry` 的整个 import 块。检查 `ReverseAttemptKey` 是否还有其他使用点，若无则一并删除其 import。

- [ ] **Step 7: 删除 scheduler 侧的四个调用点**

`dual_path/scheduler.py`：

`_activate_received_decision` 中删除整个 claim 块（约 L1491-1500）：

```python
        if result.path is PathKind.DE_READ and result.reverse_attempt_id is not None:
            if not self._path_decision_coordinator.claim_reverse_activation(
                state.request_key, result.reverse_attempt_id
            ):
                logger.info(
                    "DualPath Decode activation suppressed for request %s attempt %s: the attempt is closed",
                    request_id,
                    result.reverse_attempt_id,
                )
                return
```

同方法 except 块中删除 cancel 调用（约 L1526-1530）：

```python
            if result.path is PathKind.DE_READ and result.reverse_attempt_id is not None:
                # The claim was won but no worker work was ever published.
                self._path_decision_coordinator.cancel_reverse_publication(
                    ReverseAttemptKey(state.request_key, result.reverse_attempt_id)
                )
```

删除 `mark_reverse_work_published` 调用（约 L1555）：

```python
            self._path_decision_coordinator.mark_reverse_work_published(attempt_key, send_job.job_id)
```

`_run_job_close_action` 中删除 `mark_reverse_send_complete` 调用（约 L1884）：

```python
            self._path_decision_coordinator.mark_reverse_send_complete(attempt_key)
```

注意 `_run_job_close_action` 的 `REVERSE_SEND` 分支删掉这行后，其余逻辑（`_de_progress_deadlines.pop`、`_pending_finished_sending` 处理）全部保留。

- [ ] **Step 8: 删除 `close_registry.py`**

```bash
rm vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/close_registry.py
```

- [ ] **Step 9: 清理测试中对四个 coordinator 方法的引用**

```bash
rg -n "claim_reverse_activation|cancel_reverse_publication|mark_reverse_work_published|mark_reverse_send_complete|submit_close|_closed_reverse_records|_reverse_attempt_states|close_registry" tests/
```

按文件处理：

- `test_audit_fixes.py`：删除所有调用这些方法的测试方法（约 L254-330 区间的多个用例）。
- `test_path_decision_channel.py`：删除 close 投递与回复相关的用例。
- 若 `test_dual_path_connector.py` 或其他文件含 `PathDecisionCoordinator` 的方法名清单，同步删除这五个名字。

`test_channel_registry.py` 的 `TestRegistryMatrix` 和 `TestReplyEncoding`（除信封那个已在 Step 1 改过）全部保留——它们测的是 epoch 单调性与去重。

- [ ] **Step 10: 运行全量测试**

```bash
.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/ -q --no-header
```

预期：全绿，总数约 630。

- [ ] **Step 11: 确认 close 已无残留**

```bash
rg -n "CloseReverseAttempt|CloseReplyStatus|close_registry|claim_reverse_activation|_closed_reverse_records" vllm_ascend/ tests/
```

预期：零结果。`_closed_through_attempt_ids` 和 `STALE_CLOSED` 仍应存在（epoch 机制），不要误删。

- [ ] **Step 12: lint 与 Commit**

```bash
uvx ruff@0.14.0 check vllm_ascend/distributed/kv_transfer tests/ut/distributed/kv_transfer
uvx ruff@0.14.0 format --check vllm_ascend/distributed/kv_transfer tests/ut/distributed/kv_transfer
git add -A vllm_ascend/distributed/kv_transfer tests/ut/distributed/kv_transfer
git commit -s -m "refactor(dual_path): remove the CloseReverseAttempt control protocol

The DE control channel now carries Decision messages only. Drops the
close matrix evaluator, the closed-attempt registry, and the four
coordinator transitions that fed it, while keeping the attempt-id epoch
watermark that _register_decision_locked relies on."
```

---

## Phase B — Reverse destination hold 下线

### Task 3: 删除 `HoldLedger` 与 hold 预算

正常路径下这个 pin 是冗余的：DE_READ 请求 park 在 `WAITING_FOR_REMOTE_KVS` 期间 destination block 的 `ref_cnt` 本来就 ≥ 1，Reverse 收齐后才进 RUNNING。它唯一起作用的时刻是请求提前终止，而那正是 Phase A 已经放弃保护的场景。

**Files:**

- Modify: `dual_path/ledgers.py`
- Modify: `dual_path/scheduler.py`
- Modify: `vllm_ascend/envs.py`
- Rename: `tests/dual_path/test_hold_ledger.py` → `tests/dual_path/test_job_ledger.py`
- Modify: `tests/dual_path/test_watchdogs_limits.py`
- Modify: `tests/dual_path/test_dual_path_connector.py`、`tests/dual_path/test_split_race_cleanup.py`（两份清单）

**Interfaces:**

- Consumes: Task 2 的产物
- Produces: `ledgers.py` 只导出 `JobKind` / `JobLedger` / `JobRecord`。`JobRecord` 不再有 `affected_hold_ids` 字段，`JobLedger.create_job` 不再接受 `affected_hold_ids` 参数，签名变为 `create_job(job_kind, *, expected_worker_count, reverse_attempt_key=None) -> JobRecord`。`DualPathConnectorScheduler` 不再有 `_hold_ledger` / `_reverse_destination_holds` / `_max_held_recovery_blocks` / `_max_recovery_records` 字段，不再有 `_ensure_reverse_destination_hold` / `_check_hold_budget` 方法和 `DualPathHoldBudgetExceededError` 异常。

---

- [ ] **Step 1: 写失败的测试——Reverse destination 不再被 pin**

在 `tests/dual_path/test_hold_ledger.py` 中，把 `TestReverseDestinationHoldLifecycle::test_de_read_holds_reverse_destination_before_delivery` 替换为下面这个反向断言的用例，并把类名改为 `TestReverseDestinationNotPinned`：

```python
class TestReverseDestinationNotPinned:
    def test_de_read_reverse_destination_blocks_are_never_pinned(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _make_request(
            target_tokens=48,
            prompt_tokens=49,
            local_tokens=16,
            store_tokens=32,
            destination_block_ids=[[20, 21, 22, 23]],
        )

        assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
        scheduler.update_state_after_alloc(request, _blocks(([70, 71],)), 16)

        # The Decision still ships, but the Reverse destination slice
        # [L_PE, K_DE) = [16, 32), which lives on block 71, is no longer pinned.
        assert coordinator.submit.call_count == 1
        assert pool.blocks[71].ref_cnt == 0
        assert not hasattr(scheduler, "_hold_ledger")
        assert not hasattr(scheduler, "_reverse_destination_holds")
```

签名参考（本文件已有的 import 与 fixture，不要改动）：`pe_scheduler_factory(path=PathKind.PE_READ, *, world_size=1, pool=None, policy=None)` 返回 `(scheduler, coordinator)`；`_make_request` 全部是关键字参数；`_blocks` 与 `_make_request` 从 `test_pe_read_forward.py` import，`make_block_pool` 从 `conftest.py` import。

- [ ] **Step 2: 运行，确认失败**

```bash
.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/test_hold_ledger.py -q --no-header
```

预期：FAIL，`ref_cnt` 因为 hold 的 `touch()` 比 before 大 1，且 `_hold_ledger` 仍存在。

- [ ] **Step 3: 从 `ledgers.py` 删除 hold 三件套**

`dual_path/ledgers.py` 删除 `HoldKind`、`HoldRecord`、`HoldLedger` 三个定义，`__all__` 收缩为：

```python
__all__ = [
    "JobKind",
    "JobLedger",
    "JobRecord",
]
```

`JobRecord` 删除 `affected_hold_ids` 字段，`JobLedger.create_job` 删除同名参数及其传递。删除 `BlockPool` 的 `TYPE_CHECKING` import（`JobLedger` 不碰 block pool）。

模块 docstring 替换为：

```python
"""Completion-job ledger for DualPath Stage-2.

Both roles aggregate all-worker completion proofs through this ledger: the PE
role gates a parked DE_READ request on its Reverse completion job, and the DE
role gates the source blocks of a finishing request on its Reverse-send job.
Reports are counted exactly once per worker and capped at the expected count,
so a duplicated report can never close a job early or twice.

No KV block is pinned here. Both directions follow the parent's immediate-free
semantics: Forward is the ordinary Layerwise push, and an aborted Reverse
destination is released with the request.
"""
```

`JobLedger.record_failure` 的 docstring 里去掉 "without releasing holds"，改为 `"""Close the job as failed; True iff newly closed."""`。

- [ ] **Step 4: 从 scheduler 删除 hold 获取与预算**

`dual_path/scheduler.py`：

删除 `DualPathHoldBudgetExceededError` 类定义（约 L186）、`_check_hold_budget` 方法（约 L1127-1141）、`_ensure_reverse_destination_hold` 方法（约 L1143-1165）。

`__init__` 中删除：

```python
        self._hold_ledger = HoldLedger()
        self._reverse_destination_holds: dict[str, int] = {}
        self._max_held_recovery_blocks: int = ascend_envs.VLLM_ASCEND_DUALPATH_MAX_HELD_RECOVERY_BLOCKS
        self._max_recovery_records: int = ascend_envs.VLLM_ASCEND_DUALPATH_MAX_RECOVERY_RECORDS
```

以及环境变量校验元组里对应的两行。import 中删除 `HoldKind`、`HoldLedger`。

`_update_prefill_state_after_alloc` 中删除整个 hold 获取块（约 L1001-1013），使 `_deliver_prefill_decision` 之前只剩 parking 逻辑：

```python
        result = self._prefill_path_results.get(request_id)
        if result is None:
            return
        if (
            result.path is PathKind.DE_READ
            and request_id in self._prefill_reverse_plans
            and request_id not in self._prefill_vacuous_reverse_request_ids
        ):
            # Parking in WAITING_FOR_REMOTE_KVS: the I4 gate only admits the
            # Reverse completion job of exactly this attempt.
            self._waiting_reverse_attempt_ids.setdefault(
                request_id,
                ReverseAttemptKey(result.request_key, result.reverse_attempt_id),
            )
        self._deliver_prefill_decision(request_id, request=request, blocks=blocks)
```

注意保留紧邻其上的那次 `result = self._prefill_path_results.get(request_id)` 重取——`_activate_de_read_path` 可能已使其失效。

- [ ] **Step 5: 清理 hold 释放点与 `affected_hold_ids` 引用**

`dual_path/scheduler.py` 的 `_close_reverse_completion_job` 删除释放块，剩下：

```python
        del self._waiting_reverse_attempt_ids[request_id]
        self._recovery_deadlines.pop(request_id, None)
        self._job_ledger.discard(job.job_id)
        return {request_id}
```

`_ensure_reverse_destination_hold` 已删，两处 `create_job(JobKind.REVERSE_COMPLETION, ...)` 调用无需改动（它们本来就没传 `affected_hold_ids`，是事后赋值的）。

`_aggregate_worker_job_facts` 中把失败日志里的 hold 措辞改掉：

```python
                logger.error(
                    "DualPath job %s (kind=%s) reported failed; the owning request is failed closed",
                    job_id,
                    job.job_kind.value,
                )
```

`_sweep_prefill_recovery_watchdogs` 的注释改为：

```python
        # Expiry fails the request through the control-failure path.
```

- [ ] **Step 6: 全局搜残留**

```bash
rg -n "hold|Hold|HOLD" vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/
```

逐条确认只剩自然语言里的 "hold"（例如 `_delay_free_for_connector` 注释里的 "holds a request back"）。

- [ ] **Step 7: 删除两个预算环境变量**

`vllm_ascend/envs.py` 删除 `VLLM_ASCEND_DUALPATH_MAX_HELD_RECOVERY_BLOCKS` 和 `VLLM_ASCEND_DUALPATH_MAX_RECOVERY_RECORDS` 两项及其注释。

- [ ] **Step 8: 重组测试文件**

```bash
git mv tests/ut/distributed/kv_transfer/dual_path/test_hold_ledger.py tests/ut/distributed/kv_transfer/dual_path/test_job_ledger.py
```

在新文件中删除 `TestHoldLedger` 整个类，保留 `TestJobLedger` 全部、Step 1 新写的 `TestReverseDestinationNotPinned`、以及 `test_expected_worker_count_from_world_size` / `test_worker_metadata_aggregate_rejects_foreign_type`。`_ledgers()` helper 保留（`TestJobLedger` 仍在用它做动态 import）。

把文件 docstring 改为 `"""Stage-2 W2: completion-job ledger accounting and the no-pinning contract."""`。

`TestJobLedger::test_failure_closes_the_job_without_releasing_holds` 改名为 `test_failure_closes_the_job_as_failed`，删掉其中对 `affected_hold_ids` 的构造与断言。

`test_forward_source_blocks_are_never_pinned` 保留但要改断言——它目前有一行 `assert scheduler._hold_ledger.unreleased_count() == 0` 引用了已删除的字段，改为：

```python
        assert not hasattr(scheduler, "_hold_ledger")
        assert scheduler._job_ledger.open_count() == 0
        assert pool.blocks[11].ref_cnt == 0
        assert pool.blocks[12].ref_cnt == 0
```

- [ ] **Step 9: 修 `test_watchdogs_limits.py`**

删除 `TestHoldPressureLimits` 整个类（两个用例都在测已删除的预算）。

`TestWatchdogs::test_pe_recovery_watchdog_expiry_fails_request_without_releasing_holds` 改名为 `test_pe_recovery_watchdog_expiry_fails_request_through_control_failure`，删除其中对 `_hold_ledger` / `held_block_count()` 的断言，保留对 `control_failures` 与 `_prefill_invalid_request_ids` 的断言。

`test_env_vars_registered_with_defaults` 的清单里去掉两个预算变量。

`TestCleanupRetention::test_logical_cleanup_removes_forward_state_as_unit` 中去掉对 hold 的断言。

- [ ] **Step 10: 清理其余测试文件中散落的 hold 引用**

```bash
rg -n "_hold_ledger|_reverse_destination_holds|HoldKind|HoldLedger|affected_hold_ids|held_block_count" tests/
```

逐个处理（这些是 Step 8/9 之外的全部命中）：

- `test_forward_abort_release.py:31`：`assert scheduler._hold_ledger.unreleased_count() == 0` 改为 `assert not hasattr(scheduler, "_hold_ledger")`。
- `test_topology_validation.py:70` 与 `:84`：`assert scheduler._hold_ledger._records == {}` 整行删除。两处所在的用例对 `_prefill_invalid_request_ids` 与 coordinator 调用次数的断言已经覆盖了原意图。
- `test_resume_admission.py:95/104` 与 `:131/136`：这两组断言的意图是"重入 admission 不新增记录"，把计数对象从 hold 换成 job：

  ```python
      job_records_before = len(scheduler._job_ledger._records)
      ...
      assert len(scheduler._job_ledger._records) == job_records_before
  ```

- `test_audit_fixes.py`：删除顶部 `HoldKind` / `HoldLedger` 的 import；删除 L296 附近那个直接构造 `HoldLedger` 的用例（它验证的是"失败 job 的 hold 不被提前回收"，随 hold 一起消失）；L212-236 两处把 `_reverse_destination_holds` / `_hold_ledger.get(...)` 的断言改为对应的 job 记录断言（失败 job 保留、正常关闭后回收）。

- [ ] **Step 11: 同步两份契约清单**

`test_dual_path_connector.py` 的 `dual_path_scheduler_fields` 删除：

```python
            "_hold_ledger",
            "_reverse_destination_holds",
            "_max_held_recovery_blocks",
            "_max_recovery_records",
```

`test_dual_path_connector.py` 和 `test_split_race_cleanup.py` 两份方法清单各删除：

```python
        "_ensure_reverse_destination_hold",
        "_check_hold_budget",
```

- [ ] **Step 12: 全量测试 + lint**

```bash
.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/ -q --no-header
uvx ruff@0.14.0 check vllm_ascend/distributed/kv_transfer vllm_ascend/envs.py tests/ut/distributed/kv_transfer
uvx ruff@0.14.0 format --check vllm_ascend/distributed/kv_transfer vllm_ascend/envs.py tests/ut/distributed/kv_transfer
```

预期：全绿，总数约 615。

> **中间状态提示**：Step 3 删掉 `HoldLedger` 之后、Step 8-11 修完测试之前，测试是红的（大量 `AttributeError: HoldLedger`）。这是预期的，不要在 Step 3-7 之间反复跑全量测试试图让它变绿。

- [ ] **Step 13: Commit**

```bash
git add -A vllm_ascend/distributed/kv_transfer vllm_ascend/envs.py tests/ut/distributed/kv_transfer
git commit -s -m "refactor(dual_path): stop pinning Reverse destination blocks

The hold was redundant on the normal path, where a parked DE_READ request
already owns its destination blocks until the Reverse completes. Its only
effect was on early termination, which now follows the parent's
immediate-free semantics. ledgers.py keeps the completion-job ledger only."
```

---

## Phase C — 连带修复与收尾

### Task 4: 回收已关闭的 job 记录

Task 3 删掉了 `_max_recovery_records` 预算，而失败的 job 记录目前从不回收（`record_failure` 把 job 置为 closed，但没有任何路径调用 `discard`），孤儿 attempt 的 `REVERSE_SEND` 记录同样如此。预算是这两处泄漏此前唯一的兜底，删了预算就必须补上回收，否则长跑进程会无界增长。

**Files:**

- Modify: `dual_path/scheduler.py`
- Test: `tests/dual_path/test_job_ledger.py`

**Interfaces:**

- Consumes: Task 3 的产物（`JobLedger.discard(job_id) -> bool`，仅对 `closed=True` 的记录生效）
- Produces: 无新增公开符号；`_release_scheduler_request_state` 额外承担关闭态 job 记录的回收。

---

- [ ] **Step 1: 写失败的测试**

在 `tests/dual_path/test_job_ledger.py` 末尾追加：

```python
class TestJobRecordReclamation:
    @staticmethod
    def _admit_de_read(pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _make_request(
            target_tokens=48,
            prompt_tokens=49,
            local_tokens=16,
            store_tokens=32,
            destination_block_ids=[[20, 21, 22, 23]],
        )
        assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
        scheduler.update_state_after_alloc(request, _blocks(([70, 71],)), 16)
        return scheduler, request

    def test_failed_completion_job_record_is_reclaimed_with_the_request(self, pe_scheduler_factory):
        scheduler, request = self._admit_de_read(pe_scheduler_factory)
        job_id = scheduler._prefill_pending_reverse_receive_bindings[
            request.request_id
        ].reverse_completion_job_id
        assert scheduler._job_ledger.record_failure(job_id) is True

        scheduler._release_scheduler_request_state(request)

        assert scheduler._job_ledger.get(job_id) is None

    def test_open_completion_job_record_survives_request_cleanup(self, pe_scheduler_factory):
        scheduler, request = self._admit_de_read(pe_scheduler_factory)
        job_id = scheduler._prefill_pending_reverse_receive_bindings[
            request.request_id
        ].reverse_completion_job_id

        scheduler._release_scheduler_request_state(request)

        assert scheduler._job_ledger.get(job_id) is not None
```

第二个用例锁定的是"开着的 job 不能被 cleanup 顺手删掉"——它还可能收到 worker 报告，`JobLedger.discard` 对未关闭的记录返回 `False` 正是为此。

- [ ] **Step 2: 运行，确认第一个用例失败**

```bash
.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/test_job_ledger.py::TestJobRecordReclamation -q --no-header
```

预期：`test_failed_completion_job_record_is_reclaimed_with_the_request` FAIL（记录仍在），另一个 PASS。

- [ ] **Step 3: 在请求清理中回收 PE 侧关闭态 job**

`dual_path/scheduler.py` 的 `_release_scheduler_request_state`，prefill 分支中把这一行：

```python
            self._prefill_pending_reverse_receive_bindings.pop(request_id, None)
```

替换为：

```python
            released_binding = self._prefill_pending_reverse_receive_bindings.pop(request_id, None)
            if released_binding is not None:
                # A closed record can no longer receive worker reports; an open
                # one still can, and discard() refuses it.
                self._job_ledger.discard(released_binding.reverse_completion_job_id)
```

- [ ] **Step 4: 回收 DE 侧被 refresh 覆盖的孤儿 send job**

同方法中，保留其上的 `self._de_progress_deadlines.pop(request_id, None)`，把紧随其后的 `latest_attempt` 处理块（从 `latest_attempt = self._latest_reverse_attempt_ids.get(request_id)` 到 `self._job_ledger.discard(send_job.job_id)`）整体替换为遍历该请求全部 attempt 的版本：

```python
        if state is not None:
            for attempt_key in [
                key for key in self._reverse_send_job_ids if key.request_key == state.request_key
            ]:
                send_job = self._job_ledger.get(self._reverse_send_job_ids[attempt_key])
                if send_job is None or send_job.closed:
                    self._reverse_send_job_ids.pop(attempt_key, None)
                if send_job is not None and send_job.closed:
                    self._job_ledger.discard(send_job.job_id)
            latest_attempt = self._latest_reverse_attempt_ids.get(request_id)
            if latest_attempt is not None:
                latest_key = ReverseAttemptKey(state.request_key, latest_attempt)
                if latest_key not in self._reverse_send_job_ids:
                    self._latest_reverse_attempt_ids.pop(request_id, None)
```

原来只查 `_latest_reverse_attempt_ids` 指向的那一个 attempt，被 attempt refresh 覆盖掉的旧 attempt 记录永远留在 `_reverse_send_job_ids` 和 `JobLedger._records` 里。

- [ ] **Step 5: 运行测试确认通过**

```bash
.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/test_job_ledger.py -q --no-header
```

预期：全绿。

- [ ] **Step 6: 全量测试**

```bash
.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/ -q --no-header
```

预期：全绿。特别留意 `test_de_read_recovery.py` 和 `test_i4_gate.py`——它们覆盖 attempt refresh 场景，是 Step 4 的主要风险面。

- [ ] **Step 7: Commit**

```bash
git add -A vllm_ascend/distributed/kv_transfer tests/ut/distributed/kv_transfer
git commit -s -m "fix(dual_path): reclaim closed job records with their request

Failed jobs and send jobs superseded by an attempt refresh were never
discarded, and the recovery-record budget that used to bound them is gone.
Request cleanup now discards every closed record it owns; open records are
still refused by discard() because they can receive worker reports."
```

---

### Task 5: 删除 `_is_reverse_send_complete` 死代码

生产代码零调用（`mark_reverse_send_complete` 在 Task 2 中已删，它是唯一的语义消费方），只有测试和两份方法清单在引用。

**Files:**

- Modify: `dual_path/scheduler.py`
- Modify: `tests/dual_path/test_de_reverse_send_proof.py`
- Modify: `tests/dual_path/test_dual_path_connector.py`、`tests/dual_path/test_split_race_cleanup.py`

**Interfaces:**

- Consumes: Task 4 的产物
- Produces: `DualPathConnectorScheduler` 不再有 `_is_reverse_send_complete`。

---

- [ ] **Step 1: 确认零调用**

```bash
rg -n "_is_reverse_send_complete" vllm_ascend/ tests/
```

预期：`vllm_ascend/` 下只有定义本身；其余全在 `tests/`。若 `vllm_ascend/` 下出现调用点，**停止本 Task 并报告**——说明前面的 Task 改动与预期不符。

- [ ] **Step 2: 把测试断言换成直接读 job ledger**

`tests/dual_path/test_de_reverse_send_proof.py` 中，把形如：

```python
    assert scheduler._is_reverse_send_complete(attempt_key) is False
```

替换为等价的直接断言：

```python
    send_job = scheduler._job_ledger.get(scheduler._reverse_send_job_ids[attempt_key])
    assert not (send_job.closed and not send_job.failed)
```

`is True` 的那处（`test_job_close_makes_sender_complete_true`）替换为：

```python
    send_job = scheduler._job_ledger.get(scheduler._reverse_send_job_ids[attempt_key])
    assert send_job.closed and not send_job.failed
```

并把该用例改名为 `test_job_close_marks_the_send_job_closed_and_not_failed`。`test_audit_fixes.py` 中若还残留一处 `_is_reverse_send_complete` 断言，同样处理。

- [ ] **Step 3: 从两份方法清单删除**

`test_dual_path_connector.py` 和 `test_split_race_cleanup.py` 各删除一行 `"_is_reverse_send_complete",`。

- [ ] **Step 4: 删除方法定义**

`dual_path/scheduler.py` 删除 `_is_reverse_send_complete` 方法完整定义（约 L1922-1929）。

- [ ] **Step 5: 全量测试 + Commit**

```bash
.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/ -q --no-header
git add -A vllm_ascend/distributed/kv_transfer tests/ut/distributed/kv_transfer
git commit -s -m "refactor(dual_path): drop the unused _is_reverse_send_complete helper

Its only semantic consumer was the closed-attempt registry, removed with
the close protocol. Tests now read the job ledger directly."
```

---

### Task 6: 同步设计文档

**Files:**

- Modify: `.specs/dual-path-stage2-preemption/2026-08-11-dual-path-stage2-preemption-safety-design.md`
- Modify: `.specs/dual-path-stage2-preemption/README.md`

**Interfaces:**

- Consumes: Task 1-5 的全部产物
- Produces: 无代码符号。

---

- [ ] **Step 1: 定位需要修订的章节**

```bash
rg -n "CloseReverseAttempt|SAFE|NOT_SAFE|hold|Hold|I8|close matrix" .specs/dual-path-stage2-preemption/
```

- [ ] **Step 2: 改写而非删除**

不要删掉这些章节，改写成"已放弃的设计 + 放弃理由"。在文档中相应位置写入：

```markdown
> **2026-08-14 修订：CloseReverseAttempt 协议与 Reverse destination hold 已下线。**
>
> 二者服务的都是同一个场景：PE 侧请求在 Reverse 完成前异常终止（abort 或
> recovery watchdog 超时），此时 DE 可能仍在写 PE 的 destination block。
> 现决定该场景与 main 分支保持一致的语义——立即释放，不等远程安全证明。
>
> 保留的部分：`reverse_attempt_id`（取值 `num_preemptions`）作为控制面
> epoch，`_register_decision_locked` 的去重与单调性，以及 I4 gate。控制面
> 投递是 at-least-once 且多个 attempt 并发投递，执行端仍须据此拒绝过期指令。
```

- [ ] **Step 3: 更新 README 的模块清单**

`.specs/dual-path-stage2-preemption/README.md` 中删除 `close_registry.py` 条目，把 `ledgers.py` 的描述改为只含 `JobLedger`。

- [ ] **Step 4: markdownlint 与 Commit**

```bash
bash format.sh ci
git add -A .specs/dual-path-stage2-preemption
git commit -s -m "docs(dual_path): record the retirement of the close protocol and Reverse hold"
```

若 `format.sh ci` 修改了文件，重新 `git add` 后再 commit。

---

## 完成后的验收

- [ ] `rg -n "CloseReverseAttempt|close_registry|HoldLedger|HoldKind|_reverse_destination_holds" vllm_ascend/ tests/` 返回空
- [ ] `rg -n "_closed_through_attempt_ids|STALE_CLOSED|reverse_attempt_id" vllm_ascend/` 仍有结果（epoch 机制完好）
- [ ] `.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/ -q --no-header` 全绿
- [ ] `.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/ -q --no-header -p no:randomly` 同样全绿（排除用例间耦合）
- [ ] `uvx ruff@0.14.0 check` 与 `format --check` 干净
- [ ] `git log --oneline` 显示 6 个签名 commit

## 已知的行为变更（需在 PR 描述中说明）

1. PE 侧请求 abort 或 recovery watchdog 超时时，Reverse destination block 立即释放。若 DE 此时仍在发送，会写入已重分配的块，可能污染其他请求。这是与 main 一致的风险等级（main 的 P→D 方向存在同构问题），但 DualPath 多了 D→P 一个方向。
2. 不再有 hold 压力限制，`VLLM_ASCEND_DUALPATH_MAX_HELD_RECOVERY_BLOCKS` 与 `VLLM_ASCEND_DUALPATH_MAX_RECOVERY_RECORDS` 两个环境变量移除，设置它们不再有任何效果。
3. `VLLM_ASCEND_DUALPATH_CLOSE_RETRY_BACKOFF_S` 移除。
4. DE 控制端点不再接受 `CloseReverseAttempt` 消息；收到会记一条 warning 并丢弃。跨版本混部时旧 PE 发来的 close 会得不到回复，其 `_deliver_close` 将在重试耗尽后失败——**不支持新旧版本混部**。

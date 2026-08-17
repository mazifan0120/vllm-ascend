# DualPath Connector Module Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the DualPath Scheduler and Worker implementations out of
`connector.py` while preserving the legacy public import path and all runtime
behavior.

**Architecture:** `scheduler.py` and `worker.py` own their process-local
classes, state types, helpers, and implementation dependencies. `connector.py`
imports both classes, constructs them behind `DualPathConnector`, and re-exports
the four compatibility names. Neither owning module imports the facade or the
other owning module.

**Tech Stack:** Python 3.11, vLLM KV connector v1 interfaces, pytest, Ruff,
pre-commit.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-09-dual-path-connector-module-split-design.md` exactly.
- Preserve routing, allocation, metadata, transfer, completion, failure, and shutdown behavior.
- Preserve object identity for the public legacy imports from `dual_path.connector`.
- Move implementation bodies without opportunistic cleanup or renaming.
- Patch dependencies where they are looked up after the move.
- Do not modify `.kimi-code/` or unrelated existing files.
- Do not commit, stage, push, or create a PR without explicit user authorization.

---

### Task 1: Establish the module compatibility contract

**Files:**

- Create: `tests/ut/distributed/kv_transfer/dual_path/test_module_boundaries.py`

**Interfaces:**

- Consumes: existing symbols from `dual_path.connector`.
- Produces: a failing import contract for `dual_path.scheduler` and
  `dual_path.worker` plus identity checks for facade re-exports.

- [ ] **Step 1: Run the existing DualPath unit-test directory as the baseline**

Run:

```bash
.venv/bin/python -m pytest -q tests/ut/distributed/kv_transfer/dual_path
```

Expected: the current suite passes before production files move.

- [ ] **Step 2: Add the Scheduler compatibility test**

```python
def test_scheduler_symbols_have_one_definition_and_legacy_exports():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler import (
        DecodeKVSnapshot,
        DualPathConnectorScheduler,
    )

    assert connector.DecodeKVSnapshot is DecodeKVSnapshot
    assert connector.DualPathConnectorScheduler is DualPathConnectorScheduler
```

This catches a missing new module, a duplicate compatibility wrapper, or a
facade that exports a different class object.

- [ ] **Step 3: Add the Worker compatibility test**

```python
def test_worker_symbol_has_one_definition_and_legacy_export():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.worker import (
        DualPathConnectorWorker,
    )

    assert connector.DualPathConnectorWorker is DualPathConnectorWorker
```

- [ ] **Step 4: Verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/ut/distributed/kv_transfer/dual_path/test_module_boundaries.py
```

Expected: collection fails with `ModuleNotFoundError` for
`dual_path.scheduler` or `dual_path.worker` because neither module exists.

---

### Task 2: Move Scheduler ownership into `scheduler.py`

**Files:**

- Create: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- Modify: Scheduler-focused tests under
  `tests/ut/distributed/kv_transfer/dual_path/`

**Interfaces:**

- Consumes: `DualPathConfig`, DualPath metadata, decision/channel types,
  `KVPoolSchedulerAdapter`, and `MooncakeLayerwiseConnectorScheduler`.
- Produces: `DecodeKVSnapshot` and `DualPathConnectorScheduler` for facade
  construction and compatibility export.

- [ ] **Step 1: Create `scheduler.py` with Scheduler-owned imports and state**

Move these definitions without semantic changes:

```python
@dataclass(frozen=True)
class DecodeKVSnapshot: ...

class _DecodeDecisionStatus(str, Enum): ...

@dataclass(slots=True)
class DecodePathDecisionState: ...

class _AdmissionLookup(NamedTuple): ...

def _decode_ready_token_count(num_tokens: int) -> int: ...
def _classify_store_hit(...): ...
def _expected_prefill_token_end(...): ...
def _is_open_decision_status(...): ...

class DualPathConnectorScheduler(MooncakeLayerwiseConnectorScheduler): ...
```

Include `__all__ = ["DecodeKVSnapshot", "DualPathConnectorScheduler"]`.

- [ ] **Step 2: Make the facade import Scheduler symbols**

In `connector.py`:

```python
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler import (
    DecodeKVSnapshot,
    DualPathConnectorScheduler,
)
```

Remove the moved definitions and Scheduler-only imports. Keep the Worker and
facade definitions unchanged at this checkpoint.

- [ ] **Step 3: Update Scheduler implementation patch targets**

Change test patches for the following lookup names from `dual_path.connector`
to `dual_path.scheduler`:

```text
KVPoolSchedulerAdapter
PathDecisionCoordinator
PathDecisionDecider
derive_decode_control_port
get_ip
logger
time
```

Tests that inspect `_DecodeDecisionStatus` import
`dual_path.scheduler as scheduler_module`. Metadata and decision types are
imported from their owning modules instead of through the facade.

- [ ] **Step 4: Verify the Scheduler half is green**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/ut/distributed/kv_transfer/dual_path/test_module_boundaries.py::test_scheduler_symbols_have_one_definition_and_legacy_exports \
  tests/ut/distributed/kv_transfer/dual_path/test_decode_scheduler.py \
  tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py \
  tests/ut/distributed/kv_transfer/dual_path/test_decode_admission_integration.py \
  tests/ut/distributed/kv_transfer/dual_path/test_pe_read_forward.py \
  tests/ut/distributed/kv_transfer/dual_path/test_pe_read_forward_integration.py
```

Expected: all selected tests pass and the legacy Scheduler import is
object-identical to the new definition.

- [ ] **Step 5: Verify the Worker contract remains RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/ut/distributed/kv_transfer/dual_path/test_module_boundaries.py::test_worker_symbol_has_one_definition_and_legacy_export
```

Expected: `ModuleNotFoundError` for `dual_path.worker`.

---

### Task 3: Move Worker ownership into `worker.py`

**Files:**

- Create: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- Modify: Worker-focused tests under
  `tests/ut/distributed/kv_transfer/dual_path/`

**Interfaces:**

- Consumes: `DualPathConfig`, DualPath transfer metadata,
  `KVPoolWorkerAdapter`, and `MooncakeLayerwiseConnectorWorker`.
- Produces: `DualPathConnectorWorker` for facade construction and compatibility
  export.

- [ ] **Step 1: Create `worker.py` with Worker-owned imports and state**

Move these definitions without semantic changes:

```python
class _SplitPhase(str, Enum): ...

@dataclass(slots=True)
class _SplitTracker: ...

class DualPathConnectorWorker(MooncakeLayerwiseConnectorWorker): ...
```

Include `__all__ = ["DualPathConnectorWorker"]`.

- [ ] **Step 2: Make the facade import the Worker symbol**

In `connector.py`:

```python
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.worker import (
    DualPathConnectorWorker,
)
```

Remove the moved definitions and Worker-only imports. The resulting file keeps
only the facade, its direct vLLM dependencies, metadata construction, and the
three compatibility imports.

- [ ] **Step 3: Update Worker implementation patch targets**

Change test patches for `KVPoolWorkerAdapter` from `dual_path.connector` to
`dual_path.worker`. Tests that need Worker-private state import
`dual_path.worker as worker_module`.

- [ ] **Step 4: Verify GREEN for the complete compatibility contract**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/ut/distributed/kv_transfer/dual_path/test_module_boundaries.py \
  tests/ut/distributed/kv_transfer/dual_path/test_bidirectional_registration.py \
  tests/ut/distributed/kv_transfer/dual_path/test_forward_receive_binding.py \
  tests/ut/distributed/kv_transfer/dual_path/test_split_integration.py \
  tests/ut/distributed/kv_transfer/dual_path/test_split_lifecycle.py \
  tests/ut/distributed/kv_transfer/dual_path/test_split_race_cleanup.py \
  tests/ut/distributed/kv_transfer/dual_path/test_split_failure_provenance.py
```

Expected: all selected tests pass.

---

### Task 4: Finalize the facade and run regression verification

**Files:**

- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- Verify: all files changed by Tasks 1-3

**Interfaces:**

- Consumes: the Scheduler and Worker classes from their owning modules.
- Produces: the stable `DualPathConnector` facade and its legacy public export
  surface.

- [ ] **Step 1: Define the explicit facade export list**

```python
__all__ = [
    "DecodeKVSnapshot",
    "DualPathConnector",
    "DualPathConnectorScheduler",
    "DualPathConnectorWorker",
]
```

Confirm `connector.py` has no Scheduler/Worker implementation helpers or
adapter/control-channel imports.

- [ ] **Step 2: Compile the three ownership modules**

Run:

```bash
.venv/bin/python -m py_compile \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py
```

Expected: exit code 0.

- [ ] **Step 3: Run all DualPath tests**

```bash
.venv/bin/python -m pytest -q tests/ut/distributed/kv_transfer/dual_path
```

Expected: zero failures.

- [ ] **Step 4: Run AscendStore regression tests**

```bash
.venv/bin/python -m pytest -q tests/ut/distributed/ascend_store
```

Expected: zero failures.

- [ ] **Step 5: Run focused static checks**

```bash
git diff --check
.venv/bin/pre-commit run ruff-check --hook-stage manual --files \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/{connector,scheduler,worker}.py \
  $(rg --files tests/ut/distributed/kv_transfer/dual_path -g '*.py')
.venv/bin/pre-commit run ruff-format --hook-stage manual --files \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/{connector,scheduler,worker}.py \
  $(rg --files tests/ut/distributed/kv_transfer/dual_path -g '*.py')
.venv/bin/pre-commit run codespell --hook-stage manual --files \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/{connector,scheduler,worker}.py \
  $(rg --files tests/ut/distributed/kv_transfer/dual_path -g '*.py') \
  docs/superpowers/specs/2026-08-09-dual-path-connector-module-split-design.md \
  docs/superpowers/plans/2026-08-09-dual-path-connector-module-split.md
.venv/bin/pre-commit run typos --hook-stage manual --files \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/{connector,scheduler,worker}.py \
  $(rg --files tests/ut/distributed/kv_transfer/dual_path -g '*.py') \
  docs/superpowers/specs/2026-08-09-dual-path-connector-module-split-design.md \
  docs/superpowers/plans/2026-08-09-dual-path-connector-module-split.md
.venv/bin/pre-commit run markdownlint --hook-stage manual --files \
  docs/superpowers/specs/2026-08-09-dual-path-connector-module-split-design.md \
  docs/superpowers/plans/2026-08-09-dual-path-connector-module-split.md
```

Expected: every invoked hook passes; if `ruff-format` edits a file, rerun
affected tests and hooks.

- [ ] **Step 6: Audit the final dependency boundary**

Run:

```bash
rg -n "class DualPathConnector(Scheduler|Worker)" \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path
rg -n "from .*dual_path\.connector import" \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/{scheduler,worker}.py
git status --short
```

Expected: each implementation class has one definition in its owning module;
Scheduler and Worker do not import the facade; the status contains only the
intended refactor files plus the pre-existing untracked `.kimi-code/` directory.

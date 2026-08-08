# SPDX-License-Identifier: Apache-2.0
"""Deferred Task-07 split-runtime acceptance on real Ascend buffers."""

# NPU runbook:
# 1. Provision two injected workers (DE and PE) with distinct physical blocks.
# 2. Register real NPU KV buffers and one real TransferEngine per worker.
# 3. Implement _run_split_smoke for partial-Store and empty-Store metadata.
# 4. Remove the skip only on the supported multi-node Ascend nightly runner.
# 5. Run: pytest -sv tests/e2e/nightly/multi_node/dual_path/test_split_smoke.py

from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.skip(reason="Task-07 split smoke execution is deferred to a supported Ascend machine")


@dataclass(frozen=True, slots=True)
class SplitSmokeEvidence:
    de_block_ids: tuple[int, ...]
    pe_block_ids: tuple[int, ...]
    store_destination_blocks: tuple[int, ...]
    store_landed: bool
    reverse_mapping_exact: bool
    compute_started_after_reverse_done: bool
    inherited_forward_layer_count: int
    registered_layer_count: int
    forward_token_range: tuple[int, int]
    visible_reverse_terminals: int
    visible_forward_terminals: int
    decode_completions: int
    used_policy_result: bool
    used_coordinator_delivery: bool
    mutated_proxy: bool


def _run_split_smoke(*, partial_store: bool) -> SplitSmokeEvidence:
    pytest.fail(
        f"TODO: inject two real DualPath workers, TransferEngine, and NPU KV buffers for partial_store={partial_store}"
    )


def _assert_split_evidence(evidence: SplitSmokeEvidence, *, partial_store: bool) -> None:
    assert evidence.de_block_ids != evidence.pe_block_ids
    assert bool(evidence.store_destination_blocks) is partial_store
    assert evidence.store_landed is partial_store
    assert evidence.reverse_mapping_exact
    assert evidence.compute_started_after_reverse_done
    assert evidence.inherited_forward_layer_count == evidence.registered_layer_count
    assert evidence.forward_token_range == (64, 128)
    assert evidence.visible_reverse_terminals == 1
    assert evidence.visible_forward_terminals == 1
    assert evidence.decode_completions == 1
    assert evidence.used_policy_result is False
    assert evidence.used_coordinator_delivery is False
    assert evidence.mutated_proxy is False


def test_split_smoke_with_partial_store() -> None:
    evidence = _run_split_smoke(partial_store=True)
    _assert_split_evidence(evidence, partial_store=True)


def test_split_smoke_with_empty_store() -> None:
    evidence = _run_split_smoke(partial_store=False)
    _assert_split_evidence(evidence, partial_store=False)

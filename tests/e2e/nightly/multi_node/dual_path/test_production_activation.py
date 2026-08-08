# SPDX-License-Identifier: Apache-2.0
"""Deferred Task-08 production activation acceptance on real Ascend buffers.

NPU runbook:
1. Deploy supported PE and DE engines with a coordinated protocol-v2 upgrade;
   mixed protocol versions and rolling upgrades are not supported.
2. Configure exactly one Decode KV cache group, ``kv_load_failure_policy=fail``,
   Decode Store ``consumer_is_to_load=True`` and ``use_layerwise=False``, with
   Store/P2P block sizes and topology accepted by the existing validators.
3. Give PE and DE intentionally distinct physical KV block tables and register
   real NPU buffers and TransferEngines. Configure DualPath before the PE
   AscendStore sibling so first-positive accounting selects the intended owner.
4. Implement ``_run_production_case`` with real requests and fault injection,
   then remove the module skip only on the supported multi-node Ascend runner.
5. For every case capture evidence that PE/DE tables stay distinct; protocol-v2
   is sent only after PE allocation; Decode Store and PE Store never mix;
   ordinary Attention uses T=P; hybrid truncation uses T=R exactly once;
   inherited topology/reshard remains compatible; successful remote requests
   transition WAITING_FOR_REMOTE_KVS to WAITING; generation recomputes the
   final prompt token; and terminals/invalid blocks retain local ownership.
6. For the Store-disappears-after-probe fault, bound the request deadline and
   prove terminal failure with no fallback and no hang.
7. Run: pytest -sv
   tests/e2e/nightly/multi_node/dual_path/test_production_activation.py
"""

from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.skip(reason="Task-08 production acceptance is deferred to a supported Ascend deployment")

LogicalRange = tuple[str, str]


@dataclass(frozen=True, slots=True)
class ProductionCase:
    name: str
    route: str
    store_status: str
    store_range: LogicalRange | None
    reverse_range: LogicalRange | None
    compute_range: LogicalRange | None
    forward_range: LogicalRange | None
    protocol_v2_messages: int
    waiting_transitions: int
    success: bool
    decode_store_used: bool = False
    pe_store_used: bool = False


@dataclass(frozen=True, slots=True)
class ProductionEvidence:
    pe_block_ids: tuple[int, ...]
    de_block_ids: tuple[int, ...]
    route: str
    store_status: str
    store_range: LogicalRange | None
    reverse_range: LogicalRange | None
    compute_range: LogicalRange | None
    forward_range: LogicalRange | None
    protocol_v2_messages: int
    protocol_v2_messages_after_pe_allocation: int
    decode_store_used: bool
    pe_store_used: bool
    ordinary_forward_target: str
    ordinary_target_selection_count: int
    hybrid_forward_target: str
    hybrid_truncation_count: int
    inherited_topology_and_reshard_compatible: bool
    waiting_for_remote_kvs_to_waiting_count: int
    generation_resumed: bool
    final_prompt_token_recomputed: bool
    terminals_published_once: bool
    terminal_request_ownership_exact: bool
    invalid_block_request_ownership_exact: bool
    terminal_failure_published: bool
    fallback_attempts: int
    completed_before_deadline: bool


def _run_production_case(case: ProductionCase) -> ProductionEvidence:
    pytest.fail(f"TODO: execute Task-08 NPU production acceptance case {case.name!r}")


def _assert_production_evidence(evidence: ProductionEvidence, case: ProductionCase) -> None:
    assert evidence.pe_block_ids != evidence.de_block_ids
    assert evidence.route == case.route
    assert evidence.store_status == case.store_status
    assert evidence.store_range == case.store_range
    assert evidence.reverse_range == case.reverse_range
    assert evidence.compute_range == case.compute_range
    assert evidence.forward_range == case.forward_range
    assert evidence.protocol_v2_messages == case.protocol_v2_messages
    assert evidence.protocol_v2_messages_after_pe_allocation == case.protocol_v2_messages
    assert evidence.decode_store_used is case.decode_store_used
    assert evidence.pe_store_used is case.pe_store_used
    assert not (evidence.decode_store_used and evidence.pe_store_used)
    assert evidence.ordinary_forward_target == "P"
    assert evidence.ordinary_target_selection_count == 1
    assert evidence.hybrid_forward_target == "R"
    assert evidence.hybrid_truncation_count == 1
    assert evidence.inherited_topology_and_reshard_compatible
    assert evidence.waiting_for_remote_kvs_to_waiting_count == case.waiting_transitions
    assert evidence.generation_resumed is case.success
    assert evidence.final_prompt_token_recomputed is case.success
    assert evidence.terminals_published_once
    assert evidence.terminal_request_ownership_exact
    assert evidence.invalid_block_request_ownership_exact
    assert evidence.terminal_failure_published is not case.success
    assert evidence.fallback_attempts == 0
    assert evidence.completed_before_deadline


def _execute(case: ProductionCase) -> None:
    _assert_production_evidence(_run_production_case(case), case)


def test_hbm_complete_local_completion() -> None:
    _execute(ProductionCase("HBM-complete", "LOCAL", "NOT_PROBED", None, None, None, None, 0, 0, True))


def test_decode_local_store_full() -> None:
    _execute(
        ProductionCase(
            "Decode Store-full", "DE_LOCAL", "DONE", ("L_DE", "R"), None, None, None, 0, 1, True, decode_store_used=True
        )
    )


def test_forced_pe_read_with_pe_store_hit_when_l_pe_covers_k_de() -> None:
    _execute(
        ProductionCase(
            "L_PE>=K_DE with PE Store hit",
            "PE_READ",
            "NOT_COMMITTED",
            None,
            None,
            ("K_PE", "T"),
            ("L_DE", "T"),
            1,
            1,
            True,
            pe_store_used=True,
        )
    )


def test_partial_store_policy_pe_read() -> None:
    _execute(
        ProductionCase(
            "Partial Store policy PE_READ", "PE_READ", "NOT_COMMITTED", None, None, None, ("L_DE", "T"), 1, 1, True
        )
    )


def test_store_miss_policy_pe_read() -> None:
    _execute(
        ProductionCase(
            "Store miss policy PE_READ",
            "PE_READ",
            "NOT_COMMITTED",
            None,
            None,
            ("L_PE", "T"),
            ("L_DE", "T"),
            1,
            1,
            True,
        )
    )


def test_partial_store_policy_de_read_exact_ranges() -> None:
    _execute(
        ProductionCase(
            "Partial Store policy DE_READ",
            "DE_READ",
            "DONE",
            ("L_DE", "K_DE"),
            ("L_PE", "K_DE"),
            ("K_DE", "T"),
            ("K_DE", "T"),
            1,
            1,
            True,
            decode_store_used=True,
        )
    )


def test_store_miss_policy_de_read_uses_decode_hbm_reverse() -> None:
    _execute(
        ProductionCase(
            "Store miss policy DE_READ",
            "DE_READ",
            "SKIPPED",
            None,
            ("L_PE", "L_DE"),
            ("L_DE", "T"),
            ("L_DE", "T"),
            1,
            1,
            True,
        )
    )


def test_store_disappears_after_probe_fails_without_fallback_or_hang() -> None:
    _execute(
        ProductionCase(
            "Store disappears after probe",
            "DE_READ",
            "FAILED",
            ("L_DE", "K_DE"),
            ("L_PE", "K_DE"),
            ("K_DE", "T"),
            ("K_DE", "T"),
            1,
            0,
            False,
            decode_store_used=True,
        )
    )

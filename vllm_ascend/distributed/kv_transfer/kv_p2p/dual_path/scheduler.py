# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side admission and path-decision lifecycle for DualPath."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

from typing_extensions import assert_never
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.network_utils import get_ip
from vllm.v1.request import RequestStatus

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.kvpool_adapter import (
    KVPoolSchedulerAdapter,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.ledgers import (
    HoldKind,
    HoldLedger,
    JobKind,
    JobLedger,
    JobRecord,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    BlockIdGroups,
    DualPathConnectorMetadata,
    DualPathControlFailureMetadata,
    DualPathControlFailureReason,
    DualPathWorkerMetadata,
    ForwardPlan,
    ForwardReceiveBinding,
    ReversePlan,
    ReverseReceiveBinding,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionDecider,
    PathDecisionRequest,
    PathDecisionResult,
    PathDecisionValidationError,
    PathKind,
    PathPolicy,
    ReverseAttemptKey,
    RoundRobinPathPolicy,
    reverse_wire_id,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    CloseReplyStatus,
    CloseReverseAttempt,
    DecodeControlEndpoint,
    DualPathDecisionMetadata,
    PathDecision,
    PathDecisionCoordinator,
    derive_decode_control_port,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorScheduler,
    SendReqInfo,
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    LoadSpec,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


__all__ = ["DecodeKVSnapshot", "DualPathConnectorScheduler"]


@dataclass(frozen=True)
class DecodeKVSnapshot:
    """Complete admission record, created only after real allocation.

    ``local_tokens`` retains the original HBM prefix while ``store_tokens`` is
    derived from the detached ``LoadSpec``. ``final_block_ids`` is mandatory
    because the record cannot exist before vLLM allocates the final Decode blocks.
    """

    transfer_tokens: int
    local_tokens: int
    external_tokens: int
    store_load_spec: LoadSpec | None
    final_block_ids: tuple[tuple[int, ...], ...]
    allocated_blocks: KVCacheBlocks = field(compare=False, repr=False)

    @property
    def store_tokens(self) -> int:
        if self.store_load_spec is None:
            return self.local_tokens
        return self.store_load_spec.kvpool_cached_tokens


class _DecodeDecisionStatus(str, Enum):
    PENDING = "PENDING"
    COMMITTED = "COMMITTED"
    DECISION_TIMEOUT = "DECISION_TIMEOUT"
    ACTIVATION_FAILED = "ACTIVATION_FAILED"


@dataclass(slots=True)
class DecodePathDecisionState:
    decision_request: PathDecisionRequest
    request: Request
    deadline: float
    status: _DecodeDecisionStatus

    @property
    def request_key(self) -> DualPathRequestKey:
        return self.decision_request.request_key


class _AdmissionLookup(NamedTuple):
    """Detached KVPool lookup outcome bound to one admission, keyed by request id."""

    local_tokens: int
    external_tokens: int
    load_spec: LoadSpec | None


def _decode_ready_token_count(num_tokens: int) -> int:
    """Tokens Decode can receive: the last token is always computed locally."""
    return max(num_tokens - 1, 0)


class _StoreAssessment(NamedTuple):
    """Store-hit classification: fullness plus the external token count to
    advertise to vLLM for this admission."""

    store_full: bool
    external_tokens: int


def _classify_store_hit(
    spec: LoadSpec | None,
    local_tokens: int,
    ready_tokens: int,
    transfer_tokens: int,
) -> _StoreAssessment:
    """A Store hit is full only when the detached spec covers exactly up to the
    ready boundary; external tokens then stop there instead of the transfer target."""
    store_full = (
        spec is not None and spec.vllm_cached_tokens == local_tokens and spec.kvpool_cached_tokens == ready_tokens
    )
    return _StoreAssessment(store_full, (ready_tokens if store_full else transfer_tokens) - local_tokens)


def _effective_prefill_token_count(decision_request: PathDecisionRequest, need_truncate: bool) -> int:
    """Hybrid (truncating) deployments advertise one fewer token than the prompt."""
    return decision_request.target_tokens if need_truncate else decision_request.target_tokens + 1


def _is_open_decision_status(status: _DecodeDecisionStatus) -> bool:
    """Only PENDING decisions still accept an outcome or a timeout."""
    return status is _DecodeDecisionStatus.PENDING


def _validate_local_topology(vllm_config: VllmConfig) -> None:
    """Stage-2 supports exactly PP == DP == PCP == DCP == 1 (spec section 6)."""
    parallel_config = vllm_config.parallel_config
    restrictions = (
        ("pipeline_parallel_size", parallel_config.pipeline_parallel_size),
        ("data_parallel_size", parallel_config.data_parallel_size),
        ("prefill_context_parallel_size", parallel_config.prefill_context_parallel_size),
        ("decode_context_parallel_size", parallel_config.decode_context_parallel_size),
    )
    for name, value in restrictions:
        if value != 1:
            raise ValueError(f"DualPath requires {name} == 1, got {value}")


class DualPathHoldBudgetExceededError(RuntimeError):
    """A new uncommitted admission would exceed the hold-pressure budget."""


class DualPathConnectorScheduler(MooncakeLayerwiseConnectorScheduler):
    """Scheduler side of DualPathConnector.

    Decode admission combines HBM and Store lookup state, freezes final block
    allocation, receives the Prefill decision, and installs the Store,
    Reverse, and Forward plans for the selected route. Prefill evaluates the
    path decision hook from the admitted Decode facts and installs its local
    send/receive plans after allocation. Requests outside the DualPath
    admission shapes delegate to the parent unchanged.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        engine_id: str,
        dual_path_cfg: DualPathConfig,
        *,
        path_policy: PathPolicy | None = None,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, engine_id)
        self.dual_path_cfg = dual_path_cfg
        self._lookup_results: dict[str, _AdmissionLookup] = {}
        self._decode_kv_snapshots: dict[str, DecodeKVSnapshot] = {}
        self._decode_decision_states: dict[str, DecodePathDecisionState] = {}
        self._decision_timeout_seconds: int | None = None
        self._kvpool_adapter: KVPoolSchedulerAdapter | None = None
        self._accepting_decode_admission = True
        self._accepting_prefill_decisions = True
        # PE fields exist on both roles; Decode keeps a None decider and empty
        # maps rather than constructing policy state it never owns.
        self._path_decider: PathDecisionDecider | None = None
        self._prefill_request_keys: dict[str, DualPathRequestKey] = {}
        self._prefill_decision_metadata: dict[str, DualPathDecisionMetadata] = {}
        self._prefill_local_tokens: dict[str, int] = {}
        self._prefill_path_results: dict[str, PathDecisionResult] = {}
        self._prefill_forward_plans: dict[str, ForwardPlan] = {}
        # Scheduling epoch (``num_preemptions``) each Forward plan was installed
        # in, so a replay after a preemption is distinguishable from two
        # disagreeing allocations within one epoch.
        self._prefill_forward_plan_epochs: dict[str, int] = {}
        self._prefill_reverse_plans: dict[str, ReversePlan] = {}
        self._prefill_pending_reverse_receive_bindings: dict[str, ReverseReceiveBinding] = {}
        self._prefill_control_failures: dict[str, DualPathControlFailureMetadata] = {}
        self._prefill_delivery_futures: dict[str, Future[None]] = {}
        self._prefill_invalid_request_ids: set[str] = set()
        self._block_pool: BlockPool | None = None
        self._hold_ledger = HoldLedger()
        self._job_ledger = JobLedger()
        self._expected_worker_count: int = vllm_config.parallel_config.world_size
        self._reverse_destination_holds: dict[str, int] = {}
        self._pending_ordinary_release: set[str] = set()
        self._pending_finished_sending: set[str] = set()
        self._waiting_reverse_attempt_ids: dict[str, ReverseAttemptKey] = {}
        self._reverse_send_job_ids: dict[ReverseAttemptKey, int] = {}
        self._prefill_delivered_reverse_attempts: dict[str, int] = {}
        self._prefill_deferred_deliveries: set[str] = set()
        self._prefill_vacuous_reverse_request_ids: set[str] = set()
        self._latest_reverse_attempt_ids: dict[str, int] = {}
        self._pending_close_futures: dict[str, Future[CloseReplyStatus]] = {}
        self._pending_close_requests: dict[str, tuple[CloseReverseAttempt, DecodeControlEndpoint]] = {}
        self._close_retry_deadlines: dict[str, float] = {}
        self._recovery_deadlines: dict[str, float] = {}
        self._de_progress_deadlines: dict[str, float] = {}
        self._recovery_watchdog_s: int = ascend_envs.VLLM_ASCEND_DUALPATH_RECOVERY_WATCHDOG_S
        self._de_progress_watchdog_s: int = ascend_envs.VLLM_ASCEND_DUALPATH_DE_PROGRESS_WATCHDOG_S
        self._close_retry_backoff_s: int = ascend_envs.VLLM_ASCEND_DUALPATH_CLOSE_RETRY_BACKOFF_S
        self._max_held_recovery_blocks: int = ascend_envs.VLLM_ASCEND_DUALPATH_MAX_HELD_RECOVERY_BLOCKS
        self._max_recovery_records: int = ascend_envs.VLLM_ASCEND_DUALPATH_MAX_RECOVERY_RECORDS
        for env_name, env_value in (
            ("VLLM_ASCEND_DUALPATH_RECOVERY_WATCHDOG_S", self._recovery_watchdog_s),
            ("VLLM_ASCEND_DUALPATH_DE_PROGRESS_WATCHDOG_S", self._de_progress_watchdog_s),
            ("VLLM_ASCEND_DUALPATH_CLOSE_RETRY_BACKOFF_S", self._close_retry_backoff_s),
        ):
            if env_value <= 0:
                raise ValueError(f"{env_name} must be an integer greater than zero")
        for env_name, env_value in (
            ("VLLM_ASCEND_DUALPATH_MAX_HELD_RECOVERY_BLOCKS", self._max_held_recovery_blocks),
            ("VLLM_ASCEND_DUALPATH_MAX_RECOVERY_RECORDS", self._max_recovery_records),
        ):
            if env_value < 0:
                raise ValueError(f"{env_name} must be a non-negative integer")
        _validate_local_topology(vllm_config)
        if dual_path_cfg.role == "decode":
            if len(kv_cache_config.kv_cache_groups) != 1:
                raise ValueError("DualPath Decode requires exactly one KV cache group")
            if vllm_config.kv_transfer_config.kv_load_failure_policy != "fail":
                raise ValueError("DualPath Decode requires kv_load_failure_policy='fail'")
            decision_timeout_seconds = ascend_envs.VLLM_ASCEND_DUALPATH_DECISION_TIMEOUT
            if decision_timeout_seconds <= 0:
                raise ValueError("VLLM_ASCEND_DUALPATH_DECISION_TIMEOUT must be an integer greater than zero")
            self._decision_timeout_seconds = decision_timeout_seconds
            self._kvpool_adapter = KVPoolSchedulerAdapter(vllm_config, kv_cache_config)
            data_parallel_rank = vllm_config.parallel_config.data_parallel_rank
            control_port = derive_decode_control_port(
                dual_path_control_port=dual_path_cfg.dual_path_control_port,
                data_parallel_rank=data_parallel_rank,
                kv_port=vllm_config.kv_transfer_config.kv_port,
                worker_port_span=(
                    vllm_config.parallel_config.data_parallel_size * vllm_config.parallel_config.tensor_parallel_size
                ),
            )
            self._path_decision_coordinator = PathDecisionCoordinator.for_decode(
                engine_id=engine_id,
                data_parallel_rank=data_parallel_rank,
                control_endpoint=DecodeControlEndpoint(host=get_ip(), port=control_port),
            )
        else:
            policy = path_policy if path_policy is not None else RoundRobinPathPolicy()
            self._path_decider = PathDecisionDecider(policy)
            self._path_decision_coordinator = PathDecisionCoordinator.for_prefill()
        logger.info(
            "Initializing DualPath Scheduler %s (role=%s)",
            engine_id,
            dual_path_cfg.role,
        )

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self._block_pool = gpu_block_pool

    def _is_dual_path_decode_admission(self, request: Request) -> bool:
        """Decode admission applies only to Decode-role requests that arrived
        with ``do_remote_prefill is True``; everything else keeps parent behavior."""
        if not self._accepting_decode_admission or self.dual_path_cfg.role != "decode":
            return False
        params = request.kv_transfer_params
        return params is not None and params.get("do_remote_prefill") is True

    def _stage_prefill_activation_failure(
        self,
        request_id: str,
        destination_block_ids: Sequence[int],
        token_start: int,
        token_end: int,
    ) -> None:
        block_size = self.block_size[0]
        first_block = token_start // block_size
        last_block = math.ceil(token_end / block_size)
        failure = DualPathControlFailureMetadata(
            request_id=request_id,
            invalid_block_ids=tuple(destination_block_ids[first_block:last_block]),
            reason=DualPathControlFailureReason.ACTIVATION_FAILED,
        )
        self._prefill_control_failures[request_id] = failure

    def _reconcile_prefill_deliveries(self) -> None:
        """Detect failed deliveries, stage their control failures, and reclaim
        delivery futures of released requests once they complete."""
        if self._path_decider is None:
            return

        for request_id in set(self._prefill_delivery_futures):
            delivery_future = self._prefill_delivery_futures[request_id]
            if delivery_future.done():
                delivery_failed = delivery_future.cancelled() or delivery_future.exception() is not None
                if delivery_failed:
                    # A failed prior delivery cancels its deferred replacement
                    # for good: the request converges to the failure path.
                    self._prefill_deferred_deliveries.discard(request_id)
                    result = self._prefill_path_results.get(request_id)
                    if result is not None and request_id not in self._prefill_invalid_request_ids:
                        if result.path is PathKind.DE_READ:
                            binding = self._prefill_pending_reverse_receive_bindings.get(request_id)
                            if binding is not None:
                                destination_block_ids = binding.destination_block_ids[0]
                                token_start = binding.token_start
                                token_end = binding.token_end
                            else:
                                forward_plan = self._prefill_forward_plans.get(request_id)
                                token_start = self._prefill_local_tokens.get(request_id)
                                if forward_plan is None or token_start is None:
                                    continue
                                destination_block_ids = forward_plan.source_block_ids[0]
                                token_end = forward_plan.token_start
                            self._stage_prefill_activation_failure(
                                request_id,
                                destination_block_ids,
                                token_start,
                                token_end,
                            )
                            self._prefill_invalid_request_ids.add(request_id)
                            self._prefill_pending_reverse_receive_bindings.pop(request_id, None)
                        elif result.path is PathKind.PE_READ:
                            pass
                        else:
                            assert_never(result.path)
            if request_id in self._prefill_request_keys:
                # Active request: retain the future for its done-callback.
                continue
            if not delivery_future.done():
                # Released request with an in-flight delivery: reclaim once done.
                continue
            self._prefill_delivery_futures.pop(request_id, None)

    def _decide_prefill_path_for_admission(
        self,
        request: Request,
        parent_result: tuple[int, bool],
        prefill_local_tokens: int,
    ) -> tuple[int | None, bool]:
        params = request.kv_transfer_params
        if (
            not self._accepting_prefill_decisions
            or params is None
            or params.get("do_remote_decode") is not True
            or "dual_path" not in params
        ):
            return parent_result
        self._reconcile_prefill_deliveries()

        request_id = request.request_id
        if request_id in self._prefill_invalid_request_ids:
            return parent_result

        try:
            metadata = DualPathDecisionMetadata.from_dict(params["dual_path"])
        except PathDecisionValidationError as error:
            logger.error(
                "DualPath Prefill decision metadata is invalid for request %s: %s",
                request_id,
                error,
            )
            self._prefill_invalid_request_ids.add(request_id)
            return parent_result
        decision_request = metadata.decision_request
        effective_prefill_tokens = _effective_prefill_token_count(decision_request, self.need_truncate)
        if not 0 <= prefill_local_tokens <= effective_prefill_tokens:
            logger.error(
                "DualPath Prefill local tokens are invalid for request %s: expected 0 <= %s <= %s",
                request_id,
                prefill_local_tokens,
                effective_prefill_tokens,
            )
            return parent_result

        if request_id in self._prefill_delivery_futures and request_id in self._prefill_path_results:
            return self._resume_delivered_prefill_decision(
                request_id,
                metadata,
                prefill_local_tokens,
            )

        assert self._path_decider is not None
        request_key = decision_request.request_key
        try:
            result = self._path_decider.decide(
                decision_request,
                prefill_local_tokens,
                request.num_preemptions,
            )
        except PathDecisionValidationError as error:
            if request_id not in self._prefill_path_results:
                # A first-contact failure has nothing to discard: converge
                # locally instead of escaping into the vLLM scheduling loop.
                # No Result is sent and Decode converges through its decision
                # deadline.
                logger.error(
                    "DualPath Prefill decision failed for request %s: %s",
                    request_id,
                    error,
                )
                self._prefill_invalid_request_ids.add(request_id)
                return parent_result
            # The retained decision was never delivered, so the conflicting facts
            # come from an admission retry (allocation-failure retry or preemption
            # resume re-probing a changed prefix cache). Discard the uncommitted
            # decision state wholesale and decide fresh from the new facts.
            logger.warning(
                "DualPath Prefill admission facts changed for request %s before decision "
                "delivery; discarding the undelivered decision and re-deciding: %s",
                request_id,
                error,
            )
            self._discard_undelivered_prefill_decision(request_id, request_key)
            try:
                result = self._path_decider.decide(
                    decision_request,
                    prefill_local_tokens,
                    request.num_preemptions,
                )
            except PathDecisionValidationError as fresh_error:
                logger.error(
                    "DualPath Prefill fresh decision failed for request %s: %s",
                    request_id,
                    fresh_error,
                )
                self._prefill_invalid_request_ids.add(request_id)
                return parent_result

        self._prefill_request_keys[request_id] = request_key
        self._prefill_decision_metadata[request_id] = metadata
        self._prefill_local_tokens.setdefault(request_id, prefill_local_tokens)
        self._prefill_path_results.setdefault(request_id, result)

        self._log_prefill_decision(
            request_key,
            decision_request,
            prefill_local_tokens,
            effective_prefill_tokens,
            result,
        )

        if result.path is PathKind.PE_READ:
            return 0, False
        elif result.path is PathKind.DE_READ:
            reverse_tokens = decision_request.decode_store_tokens - prefill_local_tokens
            assert reverse_tokens > 0
            return reverse_tokens, True
        else:
            assert_never(result.path)

    def _resume_delivered_prefill_decision(
        self,
        request_id: str,
        metadata: DualPathDecisionMetadata,
        prefill_local_tokens: int,
    ) -> tuple[int | None, bool]:
        # The decision was already delivered: reuse the frozen PathKind and
        # never re-run eligibility or PathPolicy.
        result = self._prefill_path_results[request_id]
        self._prefill_local_tokens[request_id] = prefill_local_tokens
        if result.path is PathKind.PE_READ:
            return 0, False
        if result.path is PathKind.DE_READ:
            reverse_tokens = max(metadata.decision_request.decode_store_tokens - prefill_local_tokens, 0)
            if reverse_tokens == 0:
                # Vacuous Reverse: bypass the Reverse machinery entirely; the
                # request never parks, so no waiting-attempt entry may linger
                # for the I4 gate, and the retained old plan must not
                # reinstall one at allocation time.
                self._waiting_reverse_attempt_ids.pop(request_id, None)
                self._prefill_vacuous_reverse_request_ids.add(request_id)
            return (reverse_tokens, True) if reverse_tokens > 0 else (0, False)
        assert_never(result.path)

    def _initiate_reverse_attempt_close(self, request_id: str) -> None:
        attempt_key = self._waiting_reverse_attempt_ids.get(request_id)
        metadata = self._prefill_decision_metadata.get(request_id)
        if attempt_key is None or metadata is None or request_id in self._pending_close_requests:
            return
        close = CloseReverseAttempt(
            request_key=attempt_key.request_key,
            reverse_attempt_id=attempt_key.reverse_attempt_id,
        )
        self._pending_close_requests[request_id] = (close, metadata.decode_control_endpoint)
        self._recovery_deadlines[request_id] = time.monotonic() + self._recovery_watchdog_s
        self._submit_reverse_attempt_close(request_id)

    def _submit_reverse_attempt_close(self, request_id: str) -> None:
        close, endpoint = self._pending_close_requests[request_id]
        try:
            close_future = self._path_decision_coordinator.submit_close(endpoint, close)
        except RuntimeError as error:
            logger.error(
                "DualPath Prefill close submission failed for request %s: %s; the Reverse destination hold is retained",
                request_id,
                error,
            )
            return
        self._pending_close_futures[request_id] = close_future

    def _reconcile_reverse_attempt_closes(self) -> None:
        # The destination hold is released only on SAFE; NOT_SAFE re-arms the
        # identical retry after the close backoff, and an in-flight close or a
        # failed delivery retains the hold. The recovery watchdog bounds the
        # whole sequence.
        now = time.monotonic()
        for request_id, close_future in list(self._pending_close_futures.items()):
            if not close_future.done():
                continue
            self._pending_close_futures.pop(request_id)
            status: CloseReplyStatus | None = None
            if not close_future.cancelled() and close_future.exception() is None:
                status = close_future.result()
            if status is CloseReplyStatus.SAFE:
                self._recovery_deadlines.pop(request_id, None)
                self._close_retry_deadlines.pop(request_id, None)
                self._pending_close_requests.pop(request_id, None)
                hold_id = self._reverse_destination_holds.pop(request_id, None)
                if hold_id is not None and self._block_pool is not None:
                    self._hold_ledger.release(self._block_pool, hold_id)
            elif status is CloseReplyStatus.NOT_SAFE:
                self._close_retry_deadlines[request_id] = now + self._close_retry_backoff_s
        for request_id, retry_deadline in list(self._close_retry_deadlines.items()):
            if retry_deadline > now or request_id in self._pending_close_futures:
                continue
            self._close_retry_deadlines.pop(request_id)
            if request_id in self._pending_close_requests:
                self._submit_reverse_attempt_close(request_id)

    def _discard_undelivered_prefill_decision(self, request_id: str, request_key: DualPathRequestKey) -> None:
        # Drop every record latched by the first admission of this request. The
        # decision was never delivered (no delivery future exists), and
        # Forward/Reverse artifacts are only installed during
        # update_state_after_alloc, so none can exist here; only the
        # admission-time maps and the decider record need clearing for the
        # retry to decide as a fresh request.
        self._prefill_request_keys.pop(request_id, None)
        self._prefill_decision_metadata.pop(request_id, None)
        self._prefill_local_tokens.pop(request_id, None)
        self._prefill_path_results.pop(request_id, None)
        assert self._path_decider is not None
        self._path_decider.discard(request_key)

    def _log_prefill_decision(
        self,
        request_key: DualPathRequestKey,
        decision_request: PathDecisionRequest,
        prefill_local_tokens: int,
        effective_prefill_tokens: int,
        result: PathDecisionResult,
    ) -> None:
        # Field layout is consumed by ops log parsing; keep it stable.
        eligibility = "forced" if prefill_local_tokens >= decision_request.decode_store_tokens else "policy"
        if result.path is PathKind.PE_READ:
            store_coverage = "none"
        elif result.path is PathKind.DE_READ:
            store_coverage = (
                "partial" if decision_request.decode_store_tokens > decision_request.decode_local_tokens else "skipped"
            )
        else:
            assert_never(result.path)
        logger.info(
            "dual_path decision key=%s/%s decode_local_tokens=%s decode_store_tokens=%s prefill_local_tokens=%s "
            "decode_ready_tokens=%s target_tokens=%s eligibility=%s selected_path=%s store=%s",
            request_key.decode_engine_instance_id,
            request_key.decode_request_id,
            decision_request.decode_local_tokens,
            decision_request.decode_store_tokens,
            prefill_local_tokens,
            decision_request.target_tokens,
            effective_prefill_tokens,
            eligibility,
            result.path.value,
            store_coverage,
        )

    def _prepare_forward_plan(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        token_start: int,
    ) -> tuple[ForwardPlan, SendReqInfo] | None:
        params = request.kv_transfer_params
        assert params is not None
        metadata = self._prefill_decision_metadata[request.request_id]
        decision_request = metadata.decision_request
        result = self._prefill_path_results[request.request_id]
        if result.request_key != decision_request.request_key:
            raise PathDecisionValidationError("retained result key does not match the Decision Request key")

        token_end = request.num_prompt_tokens
        expected_token_end = _effective_prefill_token_count(decision_request, self.need_truncate)
        if token_end != expected_token_end:
            raise PathDecisionValidationError(
                f"effective Prefill token count {token_end} does not match the expected transfer target "
                f"{expected_token_end}"
            )
        if not 0 <= token_start < token_end:
            raise PathDecisionValidationError("Forward token range must satisfy 0 <= token_start < token_end")

        topology_fields = ("remote_tp_size", "remote_pcp_size", "remote_dcp_size", "remote_pp_size", "remote_dp_size")
        required_fields = (
            "remote_block_size",
            "remote_engine_id",
            "remote_host",
            "remote_port",
            *topology_fields,
        )
        missing_fields = [field_name for field_name in required_fields if field_name not in params]
        if missing_fields:
            raise PathDecisionValidationError(f"missing inherited remote fields: {missing_fields}")
        for field_name in ("remote_engine_id", "remote_host"):
            if not isinstance(params[field_name], str) or not params[field_name]:
                raise PathDecisionValidationError(f"{field_name} must be a non-empty string")
        for field_name in ("remote_port", *topology_fields):
            value = params[field_name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PathDecisionValidationError(f"{field_name} must be a positive integer")
        if params["remote_tp_size"] != self.vllm_config.parallel_config.tensor_parallel_size:
            raise PathDecisionValidationError("DualPath requires equal PE/DE tensor_parallel_size")
        for field_name in ("remote_pcp_size", "remote_dcp_size", "remote_pp_size", "remote_dp_size"):
            if params[field_name] != 1:
                raise PathDecisionValidationError(f"DualPath requires {field_name} == 1")
        if params.get("remote_cached_tokens") != decision_request.decode_local_tokens:
            raise PathDecisionValidationError("remote_cached_tokens does not match the Decision Request local prefix")

        local_block_sizes = tuple(self.block_size)
        remote_block_sizes = tuple(params["remote_block_size"])
        if not local_block_sizes or len(local_block_sizes) != len(remote_block_sizes):
            raise PathDecisionValidationError("local and remote block-size group counts must match")
        for block_size in (*local_block_sizes, *remote_block_sizes):
            if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
                raise PathDecisionValidationError("block sizes must be positive integers")
        if any(token_start % block_size != 0 for block_size in remote_block_sizes):
            raise PathDecisionValidationError("token_start must align to the Decode block size")

        destination_block_ids = tuple(tuple(group) for group in params["remote_block_ids"])
        if len(destination_block_ids) != len(remote_block_sizes):
            raise PathDecisionValidationError("destination block-table group count does not match block sizes")
        for group, block_size in zip(destination_block_ids, remote_block_sizes, strict=True):
            if len(group) * block_size < token_end:
                raise PathDecisionValidationError("destination block table does not cover token_end")

        source_block_ids = tuple(tuple(group) for group in blocks.get_block_ids())
        if len(source_block_ids) != len(local_block_sizes):
            raise PathDecisionValidationError("source block-table group count does not match block sizes")
        if any(
            len(group) * block_size < token_end
            for group, block_size in zip(source_block_ids, local_block_sizes, strict=True)
        ):
            return None

        plan = ForwardPlan(
            request_key=decision_request.request_key,
            token_start=token_start,
            token_end=token_end,
            source_block_ids=source_block_ids,
            destination_block_ids=destination_block_ids,
        )
        send_req_info = SendReqInfo(
            local_block_ids=[list(group) for group in plan.source_block_ids],
            local_transferred_tokens=plan.token_start,
            local_computed_tokens=0,
            request=request,
        )
        return plan, send_req_info

    def _may_install_forward_plan(self, request: Request, plan: ForwardPlan) -> bool:
        """Whether ``plan`` may be installed, i.e. it is new or a legal replay.

        A preemption reallocates the block table, so a pass in a later
        scheduling epoch legitimately supersedes the plan: the Forward
        direction pins nothing, so the old plan strands no resource. Within one
        epoch a differing plan means two disagreeing allocations for the same
        request, which the engine cannot resolve.
        """
        request_id = request.request_id
        existing_plan = self._prefill_forward_plans.get(request_id)
        if existing_plan is None:
            return True
        if existing_plan == plan:
            return False
        if self._prefill_forward_plan_epochs.get(request_id) == request.num_preemptions:
            raise RuntimeError(
                f"DualPath Prefill request {request_id} got a conflicting duplicate Forward plan; "
                "this is a bug and the engine cannot continue safely"
            )
        return True

    def _try_install_forward_plan(self, request: Request, blocks: KVCacheBlocks) -> None:
        request_id = request.request_id
        result = self._prefill_path_results.get(request_id)
        if result is None:
            return
        token_start = self._prefill_decision_metadata[request_id].decision_request.decode_local_tokens

        try:
            prepared = self._prepare_forward_plan(request, blocks, token_start)
        except (KeyError, PathDecisionValidationError, TypeError) as error:
            logger.error(
                "DualPath Prefill Forward plan is invalid for request %s: %s",
                request_id,
                error,
            )
            self._prefill_invalid_request_ids.add(request_id)
            self._prefill_path_results.pop(request_id, None)
            return
        if prepared is None:
            return
        plan, send_req_info = prepared

        if not self._may_install_forward_plan(request, plan):
            return
        self._prefill_forward_plans[request_id] = plan
        self._prefill_forward_plan_epochs[request_id] = request.num_preemptions
        self._reqs_need_send_layerwise[request_id] = send_req_info

    def _activate_de_read_path(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        result: PathDecisionResult,
    ) -> ReversePlan | None:
        request_id = request.request_id
        params = request.kv_transfer_params
        assert params is not None
        metadata = self._prefill_decision_metadata[request_id]
        decision_request = metadata.decision_request
        if result.request_key != decision_request.request_key:
            raise PathDecisionValidationError("retained result key does not match the Decision Request key")

        token_start = self._prefill_local_tokens[request_id]
        token_split = decision_request.decode_store_tokens
        ready_tokens = decision_request.target_tokens
        token_end = request.num_prompt_tokens
        if not 0 <= decision_request.decode_local_tokens <= token_split:
            raise PathDecisionValidationError("Decode local tokens must not exceed the DE_READ split")

        retained_reverse_plan = self._prefill_reverse_plans.get(request_id)
        replacement_attempt_id: int | None = None
        if retained_reverse_plan is not None and request.num_preemptions > retained_reverse_plan.reverse_attempt_id:
            replacement_attempt_id = request.num_preemptions
        if replacement_attempt_id is None and not token_start < token_split < ready_tokens <= token_end:
            raise PathDecisionValidationError("DE_READ token ranges must satisfy L_PE < K_DE < R <= T")

        prepared = self._prepare_forward_plan(request, blocks, token_split)
        # vLLM admission allocates only the external-token blocks (the Reverse
        # destination [L_PE, K_DE)) and then parks the request in
        # WAITING_FOR_REMOTE_KVS, so a T-covering Forward table can only appear
        # on the post-Reverse allocation. Install the Reverse artifacts from
        # the admission table and deliver immediately; the Forward plan
        # installs on that later allocation without a second delivery.
        pe_block_table = tuple(tuple(group) for group in blocks.get_block_ids())
        de_block_table = tuple(tuple(group) for group in params["remote_block_ids"])
        if any(
            len(group) * block_size < token_split
            for group, block_size in zip(pe_block_table, tuple(self.block_size), strict=True)
        ):
            raise PathDecisionValidationError("DE_READ Reverse destination block table does not cover the split point")

        if retained_reverse_plan is None:
            attempt_key = ReverseAttemptKey(result.request_key, result.reverse_attempt_id)
            completion_job = self._job_ledger.create_job(
                JobKind.REVERSE_COMPLETION,
                expected_worker_count=self._expected_worker_count,
                reverse_attempt_key=attempt_key,
            )
            binding = ReverseReceiveBinding(
                request_key=result.request_key,
                wire_request_id=reverse_wire_id(attempt_key),
                prefill_request_id=request_id,
                destination_block_ids=pe_block_table,
                token_start=token_start,
                token_end=token_split,
                reverse_attempt_id=result.reverse_attempt_id,
                prefill_local_tokens=token_start,
                reverse_completion_job_id=completion_job.job_id,
            )
            parallel_config = self.vllm_config.parallel_config
            reverse_plan = ReversePlan(
                request_key=result.request_key,
                wire_request_id=reverse_wire_id(attempt_key),
                token_start=token_start,
                token_end=token_split,
                source_block_ids=de_block_table,
                destination_block_ids=pe_block_table,
                remote_engine_id=self.engine_id,
                remote_host=self.side_channel_host,
                remote_port=self.side_channel_port,
                remote_block_sizes=tuple(self.block_size),
                remote_tp_size=parallel_config.tensor_parallel_size,
                remote_pcp_size=parallel_config.prefill_context_parallel_size,
                remote_dcp_size=parallel_config.decode_context_parallel_size,
                reverse_attempt_id=result.reverse_attempt_id,
                prefill_local_tokens=token_start,
                reverse_send_job_id=None,
            )
            existing_binding = self._prefill_pending_reverse_receive_bindings.get(request_id)
            if existing_binding is not None and existing_binding != binding:
                raise RuntimeError(
                    f"DualPath Prefill request {request_id} got a conflicting duplicate Reverse receive binding; "
                    "this is a bug and the engine cannot continue safely"
                )
            self._prefill_pending_reverse_receive_bindings[request_id] = binding
            self._prefill_reverse_plans[request_id] = reverse_plan
            self._recovery_deadlines[request_id] = time.monotonic() + self._recovery_watchdog_s
        elif replacement_attempt_id is not None and token_start < token_split:
            # Route-preserving recovery (I7): same path and frozen DE table;
            # only the attempt-local range [L_PE(new), K_DE) and the PE block
            # table change. The new attempt re-parks the request under the I4
            # gate.
            attempt_key = ReverseAttemptKey(result.request_key, replacement_attempt_id)
            completion_job = self._job_ledger.create_job(
                JobKind.REVERSE_COMPLETION,
                expected_worker_count=self._expected_worker_count,
                reverse_attempt_key=attempt_key,
            )
            binding = ReverseReceiveBinding(
                request_key=result.request_key,
                wire_request_id=reverse_wire_id(attempt_key),
                prefill_request_id=request_id,
                destination_block_ids=pe_block_table,
                token_start=token_start,
                token_end=token_split,
                reverse_attempt_id=replacement_attempt_id,
                prefill_local_tokens=token_start,
                reverse_completion_job_id=completion_job.job_id,
            )
            parallel_config = self.vllm_config.parallel_config
            reverse_plan = ReversePlan(
                request_key=result.request_key,
                wire_request_id=reverse_wire_id(attempt_key),
                token_start=token_start,
                token_end=token_split,
                source_block_ids=de_block_table,
                destination_block_ids=pe_block_table,
                remote_engine_id=self.engine_id,
                remote_host=self.side_channel_host,
                remote_port=self.side_channel_port,
                remote_block_sizes=tuple(self.block_size),
                remote_tp_size=parallel_config.tensor_parallel_size,
                remote_pcp_size=parallel_config.prefill_context_parallel_size,
                remote_dcp_size=parallel_config.decode_context_parallel_size,
                reverse_attempt_id=replacement_attempt_id,
                prefill_local_tokens=token_start,
                reverse_send_job_id=None,
            )
            self._prefill_pending_reverse_receive_bindings[request_id] = binding
            self._prefill_reverse_plans[request_id] = reverse_plan
            self._waiting_reverse_attempt_ids[request_id] = attempt_key
            self._recovery_deadlines[request_id] = time.monotonic() + self._recovery_watchdog_s

        if prepared is None:
            # Forward table does not cover T yet; it installs on the
            # post-Reverse allocation without touching the delivered Reverse.
            return self._prefill_reverse_plans[request_id]
        forward_plan, send_req_info = prepared
        if not self._may_install_forward_plan(request, forward_plan):
            return self._prefill_reverse_plans[request_id]
        self._prefill_forward_plans[request_id] = forward_plan
        self._prefill_forward_plan_epochs[request_id] = request.num_preemptions
        self._reqs_need_send_layerwise[request_id] = send_req_info
        return self._prefill_reverse_plans[request_id]

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        if not self._is_dual_path_decode_admission(request):
            parent_result = super().get_num_new_matched_tokens(request, num_computed_tokens)
            if self.dual_path_cfg.role == "prefill":
                return self._decide_prefill_path_for_admission(request, parent_result, num_computed_tokens)
            return parent_result

        request_id = request.request_id
        transfer_tokens = self._hybrid_prefill_token_count(request.num_tokens)
        ready_tokens = _decode_ready_token_count(request.num_tokens)
        local_tokens = num_computed_tokens
        if local_tokens >= ready_tokens:
            # HBM-complete: no KVPool lookup and no KVPool admission state to bind for this request.
            self._lookup_results.pop(request_id, None)
            return 0, False

        cached = self._lookup_results.get(request_id)
        if cached is not None:
            cached_spec = cached.load_spec
            cached_spec_is_current = cached_spec is None or (
                cached_spec.vllm_cached_tokens == local_tokens and cached_spec.kvpool_cached_tokens <= ready_tokens
            )
            cached_assessment = _classify_store_hit(cached_spec, local_tokens, ready_tokens, transfer_tokens)
            if (
                cached_spec_is_current
                and cached.local_tokens == local_tokens
                and cached.external_tokens == cached_assessment.external_tokens
            ):
                # Identical duplicate lookup (e.g. allocation-failure retry):
                # reuse the detached result instead of re-probing the KV pool.
                return cached.external_tokens, True
            # Changed admission facts invalidate the unbound result; discard it before
            # the fresh lookup so a failing re-probe cannot leave it behind.
            del self._lookup_results[request_id]

        assert self._kvpool_adapter is not None
        detached_spec = self._kvpool_adapter.lookup(request, local_tokens)
        external_tokens = _classify_store_hit(
            detached_spec, local_tokens, ready_tokens, transfer_tokens
        ).external_tokens
        self._lookup_results[request_id] = _AdmissionLookup(
            local_tokens=local_tokens,
            external_tokens=external_tokens,
            load_spec=detached_spec,
        )
        return external_tokens, True

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        params = request.kv_transfer_params
        if self.dual_path_cfg.role == "prefill" and params is not None and "dual_path" in params:
            self._update_prefill_state_after_alloc(request, blocks)
            return
        if not self._is_dual_path_decode_admission(request):
            # Non-selected requests delegate untouched; in particular a request
            # resumed after a completed async load (its admission flag is already
            # consumed and num_external_tokens == 0) must not re-enter the
            # duplicate-bind check against its own earlier snapshot.
            return super().update_state_after_alloc(request, blocks, num_external_tokens)
        self._bind_decode_admission_after_alloc(request, blocks, num_external_tokens)

    def _update_prefill_state_after_alloc(self, request: Request, blocks: KVCacheBlocks) -> None:
        # Prefill role: activate the retained path decision, then deliver the
        # Decision to Decode exactly once per request.
        request_id = request.request_id
        if request_id in self._prefill_invalid_request_ids:
            return
        result = self._prefill_path_results.get(request_id)
        if result is None:
            return

        try:
            if result.path is PathKind.DE_READ:
                self._activate_de_read_path(request, blocks, result)
            elif result.path is PathKind.PE_READ:
                self._try_install_forward_plan(request, blocks)
            else:
                assert_never(result.path)
        except (KeyError, PathDecisionValidationError, RuntimeError, TypeError) as error:
            logger.error(
                "DualPath Prefill activation failed locally for request %s: %s",
                request_id,
                error,
            )
            self._invalidate_prefill_activation(request, blocks, result, discard_installed_plans=False)
            return

        result = self._prefill_path_results.get(request_id)
        if result is None:
            return
        try:
            self._ensure_reverse_destination_hold(request_id, result)
        except DualPathHoldBudgetExceededError as error:
            logger.error(
                "DualPath Prefill admission rejected by the hold-pressure limit for request %s: %s",
                request_id,
                error,
            )
            self._invalidate_prefill_activation(request, blocks, result, discard_installed_plans=True)
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

    def _deliver_prefill_decision(
        self,
        request_id: str,
        *,
        request: Request | None = None,
        blocks: KVCacheBlocks | None = None,
    ) -> None:
        if request_id in self._prefill_invalid_request_ids:
            self._prefill_deferred_deliveries.discard(request_id)
            return
        result = self._prefill_path_results.get(request_id)
        if result is None:
            self._prefill_deferred_deliveries.discard(request_id)
            return
        request_key = result.request_key
        reverse_plan: ReversePlan | None = None
        if result.path is PathKind.PE_READ:
            if request_id in self._prefill_delivery_futures:
                return
            # PE_READ delivers once the Forward plan is installed.
            if request_id not in self._prefill_forward_plans:
                return
        elif result.path is PathKind.DE_READ:
            # DE_READ delivers once the Reverse artifacts are installed; its
            # Forward plan may still be deferred to the post-Reverse allocation.
            reverse_plan = self._prefill_reverse_plans.get(request_id)
            if reverse_plan is None:
                return
            if (
                request_id in self._prefill_delivery_futures
                and self._prefill_delivered_reverse_attempts.get(request_id) == reverse_plan.reverse_attempt_id
            ):
                self._prefill_deferred_deliveries.discard(request_id)
                return
        else:
            assert_never(result.path)
        existing_future = self._prefill_delivery_futures.get(request_id)
        if existing_future is not None and not existing_future.done():
            # At most one unresolved delivery Future per logical request: the
            # replacement delivery defers (never overwrites) and the next
            # build pass retries once the earlier Future resolves.
            self._prefill_deferred_deliveries.add(request_id)
            return
        metadata = self._prefill_decision_metadata[request_id]
        decision_result = result
        if reverse_plan is not None and reverse_plan.reverse_attempt_id != result.reverse_attempt_id:
            decision_result = replace(
                result,
                reverse_attempt_id=reverse_plan.reverse_attempt_id,
                prefill_local_tokens=reverse_plan.prefill_local_tokens,
            )
        decision = PathDecision(
            result=decision_result,
            reverse_plan=reverse_plan,
        )
        try:
            delivery_future = self._path_decision_coordinator.submit(
                metadata.decode_control_endpoint,
                decision,
            )
        except RuntimeError as error:
            logger.error(
                "DualPath Prefill decision submission failed locally for request %s: %s",
                request_id,
                error,
            )
            if request is not None and blocks is not None:
                self._invalidate_prefill_activation(request, blocks, result, discard_installed_plans=True)
                self._prefill_deferred_deliveries.discard(request_id)
            return
        self._prefill_delivery_futures[request_id] = delivery_future
        self._prefill_deferred_deliveries.discard(request_id)
        if reverse_plan is not None:
            self._prefill_delivered_reverse_attempts[request_id] = reverse_plan.reverse_attempt_id

        def log_delivery_failure(completed_future: Future[None]) -> None:
            if completed_future.cancelled():
                logger.warning(
                    "dual_path delivery key=%s/%s delivery_terminal=CANCELLED",
                    request_key.decode_engine_instance_id,
                    request_key.decode_request_id,
                )
                return
            error = completed_future.exception()
            if error is not None:
                logger.error(
                    "dual_path delivery key=%s/%s delivery_terminal=FAILED failure_source=DELIVERY error=%s",
                    request_key.decode_engine_instance_id,
                    request_key.decode_request_id,
                    error,
                )
                return
            logger.info(
                "dual_path delivery key=%s/%s delivery_terminal=SUCCEEDED",
                request_key.decode_engine_instance_id,
                request_key.decode_request_id,
            )

        delivery_future.add_done_callback(log_delivery_failure)

    def _check_hold_budget(self, new_block_count: int, new_record_count: int) -> None:
        """Reject a new uncommitted admission before pinning when the budget
        would be exceeded; committed holds are never evicted."""
        held_blocks = self._hold_ledger.held_block_count()
        if held_blocks + new_block_count > self._max_held_recovery_blocks:
            raise DualPathHoldBudgetExceededError(
                f"held recovery blocks {held_blocks} + {new_block_count} would exceed "
                f"the limit {self._max_held_recovery_blocks}"
            )
        open_records = self._hold_ledger.unreleased_count() + self._job_ledger.open_count()
        if open_records + new_record_count > self._max_recovery_records:
            raise DualPathHoldBudgetExceededError(
                f"open recovery records {open_records} + {new_record_count} would exceed "
                f"the limit {self._max_recovery_records}"
            )

    def _ensure_reverse_destination_hold(self, request_id: str, result: PathDecisionResult) -> None:
        # Before a DE_READ Decision is delivered, pin the PE-local Reverse
        # destination slice [L_PE, K_DE); a vacuous Reverse acquires no hold.
        if self._block_pool is None or result.path is not PathKind.DE_READ:
            return
        existing_hold_id = self._reverse_destination_holds.get(request_id)
        if existing_hold_id is not None and not self._hold_ledger.is_released(existing_hold_id):
            return
        binding = self._prefill_pending_reverse_receive_bindings.get(request_id)
        if binding is None:
            return
        block_size = self.block_size[0]
        first_block = binding.token_start // block_size
        last_block = math.ceil(binding.token_end / block_size)
        if first_block >= last_block:
            return
        destination_block_ids = tuple(binding.destination_block_ids[0][first_block:last_block])
        self._check_hold_budget(len(destination_block_ids), 1)
        hold = self._hold_ledger.acquire(self._block_pool, destination_block_ids, HoldKind.REVERSE_DESTINATION)
        self._reverse_destination_holds[request_id] = hold.hold_id
        completion_job = self._job_ledger.get(binding.reverse_completion_job_id)
        if completion_job is not None:
            completion_job.affected_hold_ids = (hold.hold_id,)

    def _invalidate_prefill_activation(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        result: PathDecisionResult,
        *,
        discard_installed_plans: bool,
    ) -> None:
        # DE_READ stages a control failure so Decode drops the advertised blocks.
        # discard_installed_plans is set only when the Decision was never
        # delivered, in which case the installed Forward/Reverse artifacts must
        # not be handed to the Worker.
        request_id = request.request_id
        if result.path is PathKind.DE_READ:
            decision_request = self._prefill_decision_metadata[request_id].decision_request
            self._stage_prefill_activation_failure(
                request_id,
                blocks.get_block_ids()[0],
                self._prefill_local_tokens[request_id],
                decision_request.decode_store_tokens,
            )
        self._prefill_invalid_request_ids.add(request_id)
        self._prefill_path_results.pop(request_id, None)
        if not discard_installed_plans:
            return
        self._prefill_reverse_plans.pop(request_id, None)
        self._prefill_pending_reverse_receive_bindings.pop(request_id, None)
        self._prefill_forward_plan_epochs.pop(request_id, None)
        if self._prefill_forward_plans.pop(request_id, None) is not None:
            self._reqs_need_send_layerwise.pop(request_id, None)

    def _bind_decode_admission_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        # Decode role: freeze the allocation into the admission snapshot, then
        # either commit a Store-full load locally or register the pending path
        # decision and notify the proxy.
        request_id = request.request_id
        allocated_block_ids = blocks.get_block_ids()
        frozen_block_ids = tuple(tuple(group) for group in allocated_block_ids)

        existing = self._decode_kv_snapshots.get(request_id)
        if existing is not None:
            if self._is_identical_duplicate_admission(request, existing, frozen_block_ids, num_external_tokens):
                return
            raise RuntimeError(
                f"DualPath request {request_id} got a conflicting duplicate admission bind; "
                "this is a bug and the engine cannot continue safely"
            )

        if num_external_tokens == 0:
            # HBM-complete admission returned (0, False): no KVPool admission
            # state to bind, and the parent must not fire its remote-prefill/metaserver flow.
            self._lookup_results.pop(request_id, None)
            return

        entry = self._lookup_results.pop(request_id, None)
        if entry is None:
            raise RuntimeError(f"DualPath request {request_id} has no admission lookup result to bind after allocation")
        local_tokens, cached_external_tokens, detached_spec = entry
        ready_tokens = _decode_ready_token_count(request.num_tokens)
        transfer_tokens = self._hybrid_prefill_token_count(request.num_tokens)
        admission_assessment = _classify_store_hit(detached_spec, local_tokens, ready_tokens, transfer_tokens)
        if not num_external_tokens == cached_external_tokens == admission_assessment.external_tokens:
            raise RuntimeError(
                f"DualPath request {request_id} external-token mismatch: vLLM allocated "
                f"{num_external_tokens}, cached lookup expected {cached_external_tokens}, "
                f"and transfer accounting expected {admission_assessment.external_tokens}"
            )
        if detached_spec is not None:
            if detached_spec.vllm_cached_tokens != local_tokens:
                raise RuntimeError(
                    f"DualPath request {request_id} detached LoadSpec local tokens "
                    f"{detached_spec.vllm_cached_tokens} != {local_tokens}"
                )
            if not local_tokens < detached_spec.kvpool_cached_tokens <= ready_tokens:
                raise RuntimeError(
                    f"DualPath request {request_id} detached LoadSpec store tokens "
                    f"{detached_spec.kvpool_cached_tokens} outside ({local_tokens}, {ready_tokens}]"
                )

        snapshot = DecodeKVSnapshot(
            transfer_tokens=transfer_tokens,
            local_tokens=local_tokens,
            external_tokens=num_external_tokens,
            store_load_spec=detached_spec,
            final_block_ids=frozen_block_ids,
            allocated_blocks=blocks,
        )
        self._decode_kv_snapshots[request_id] = snapshot

        params = request.kv_transfer_params
        assert params is not None
        if admission_assessment.store_full:
            assert detached_spec is not None
            assert self._kvpool_adapter is not None
            self._kvpool_adapter.commit_after_alloc(request, blocks, detached_spec)
            params["do_remote_prefill"] = False
            return

        self._register_pending_decode_decision(request, snapshot, allocated_block_ids, params)

    def _is_identical_duplicate_admission(
        self,
        request: Request,
        existing: DecodeKVSnapshot,
        frozen_block_ids: tuple[tuple[int, ...], ...],
        num_external_tokens: int,
    ) -> bool:
        # A repeated allocation carrying facts identical to the bound snapshot
        # is a benign retry (e.g. allocation-failure retry); a mismatch conflicts.
        ready_tokens = _decode_ready_token_count(request.num_tokens)
        transfer_tokens = self._hybrid_prefill_token_count(request.num_tokens)
        existing_assessment = _classify_store_hit(
            existing.store_load_spec, existing.local_tokens, ready_tokens, transfer_tokens
        )
        duplicate_local_tokens = (
            ready_tokens if existing_assessment.store_full else transfer_tokens
        ) - num_external_tokens
        return (
            existing.transfer_tokens == transfer_tokens
            and existing.local_tokens == duplicate_local_tokens
            and existing.external_tokens == num_external_tokens
            and existing.final_block_ids == frozen_block_ids
        )

    def _build_remote_decode_message(
        self,
        request_id: str,
        trimmed_final_block_ids: tuple[list[int], ...],
        snapshot: DecodeKVSnapshot,
        decision_metadata: DualPathDecisionMetadata,
    ) -> dict[str, Any]:
        return {
            "token_ids": [],
            "request_id": get_external_request_id(request_id),
            "do_remote_prefill": False,
            "do_remote_decode": True,
            "remote_block_ids": trimmed_final_block_ids,
            "remote_block_size": self.block_size,
            "remote_engine_id": self.engine_id,
            "remote_host": self.side_channel_host,
            "remote_port": self.side_channel_port,
            "remote_tp_size": self.vllm_config.parallel_config.tensor_parallel_size,
            "remote_pcp_size": self.vllm_config.parallel_config.prefill_context_parallel_size,
            "remote_dcp_size": self.vllm_config.parallel_config.decode_context_parallel_size,
            "remote_pp_size": self.vllm_config.parallel_config.pipeline_parallel_size,
            "remote_dp_size": self.vllm_config.parallel_config.data_parallel_size,
            "remote_cached_tokens": snapshot.local_tokens,
            "dual_path": decision_metadata.to_dict(),
        }

    def _register_pending_decode_decision(
        self,
        request: Request,
        snapshot: DecodeKVSnapshot,
        allocated_block_ids: tuple[list[int], ...],
        params: dict[str, Any],
    ) -> None:
        # Register the pending decision before notifying the proxy so a decision
        # arriving ahead of the proxy round-trip still finds its state.
        request_id = request.request_id
        coordinator = self._path_decision_coordinator
        request_key = DualPathRequestKey(coordinator.decode_engine_instance_id, request_id)
        decision_request = PathDecisionRequest(
            request_key=request_key,
            target_tokens=_decode_ready_token_count(request.num_tokens),
            decode_local_tokens=snapshot.local_tokens,
            decode_store_tokens=snapshot.store_tokens,
        )
        decision_metadata = DualPathDecisionMetadata(
            decision_request=decision_request,
            decode_control_endpoint=coordinator.decode_control_endpoint,
        )
        remote_block_ids = self._trim_hybrid_remote_block_ids(
            allocated_block_ids,
            len(request.prompt_token_ids),
        )
        message = self._build_remote_decode_message(
            request_id,
            remote_block_ids,
            snapshot,
            decision_metadata,
        )

        coordinator.register_pending(request_key)
        assert self._decision_timeout_seconds is not None
        state = DecodePathDecisionState(
            decision_request=decision_request,
            request=request,
            deadline=time.monotonic() + self._decision_timeout_seconds,
            status=_DecodeDecisionStatus.PENDING,
        )
        self._decode_decision_states[request_id] = state

        if params.get("do_virtual") is not True:
            try:
                future = self.executor.submit(
                    self._access_metaserver,
                    url=params.get("metaserver"),
                    message=message,
                )
            except RuntimeError as error:
                logger.error(
                    "DualPath Proxy submission failed for request %s: %s",
                    request_id,
                    error,
                )
            else:

                def log_proxy_failure(completed_future: Future[None]) -> None:
                    error = completed_future.exception()
                    if error is not None:
                        logger.error(
                            "DualPath Proxy request failed for request %s: %s",
                            request_id,
                            error,
                        )

                future.add_done_callback(log_proxy_failure)

        params["do_remote_prefill"] = False

    def _build_decode_control_failure(
        self,
        request_id: str,
        snapshot: DecodeKVSnapshot,
        reason: DualPathControlFailureReason,
    ) -> DualPathControlFailureMetadata:
        block_size = self.block_size[0]
        if snapshot.local_tokens % block_size != 0:
            raise RuntimeError(
                f"DualPath control failure metadata for request {request_id} requires "
                f"local_tokens ({snapshot.local_tokens}) to be aligned to block_size ({block_size})"
            )
        invalid_block_ids = snapshot.final_block_ids[0][snapshot.local_tokens // block_size :]
        if not invalid_block_ids:
            raise RuntimeError(
                f"DualPath control failure metadata for request {request_id} has no blocks "
                f"beyond local_tokens ({snapshot.local_tokens}) to invalidate"
            )
        return DualPathControlFailureMetadata(
            request_id=request_id,
            invalid_block_ids=invalid_block_ids,
            reason=reason,
        )

    def _validate_committed_decision(
        self,
        decision: PathDecision,
        state: DecodePathDecisionState,
        snapshot: DecodeKVSnapshot,
    ) -> tuple[BlockIdGroups, int, ReversePlan | None]:
        # A received Decision must reproduce the frozen admission facts exactly;
        # any mismatch fails the activation before binding construction.
        result = decision.result
        if state.request_key != result.request_key:
            raise PathDecisionValidationError("Decision key does not match the pending Decode state")
        if state.decision_request.decode_local_tokens != snapshot.local_tokens:
            raise PathDecisionValidationError("Decision local prefix does not match the frozen snapshot")
        if state.decision_request.decode_store_tokens != snapshot.store_tokens:
            raise PathDecisionValidationError("Decision Store boundary does not match the frozen snapshot")

        destination_block_ids = tuple(
            tuple(group)
            for group in self._trim_hybrid_remote_block_ids(
                snapshot.final_block_ids,
                state.decision_request.target_tokens + 1,
            )
        )
        reverse_plan: ReversePlan | None = None
        if result.path is PathKind.PE_READ:
            forward_token_start = snapshot.local_tokens
        elif result.path is PathKind.DE_READ:
            forward_token_start = snapshot.store_tokens
        else:
            assert_never(result.path)

        if result.path is PathKind.DE_READ:
            # The receive path already guarantees a DE_READ Decision carries a Reverse plan.
            reverse_plan = decision.reverse_plan
            assert reverse_plan is not None
            attempt_key = ReverseAttemptKey(state.request_key, result.reverse_attempt_id)
            if reverse_plan.wire_request_id != reverse_wire_id(attempt_key):
                raise PathDecisionValidationError("Reverse plan wire id does not match the Decode request")
            if reverse_plan.reverse_attempt_id != result.reverse_attempt_id:
                raise PathDecisionValidationError("Reverse plan attempt does not match the Decision result")
            if reverse_plan.token_end != snapshot.store_tokens:
                raise PathDecisionValidationError("Reverse plan range does not end at the frozen Store boundary")
            if reverse_plan.source_block_ids != destination_block_ids:
                raise PathDecisionValidationError("Reverse plan source does not match the advertised Decode table")
            if len(reverse_plan.remote_block_sizes) != len(self.block_size):
                raise PathDecisionValidationError("Reverse plan block-size group count does not match Decode")
            if reverse_plan.remote_tp_size != self.vllm_config.parallel_config.tensor_parallel_size:
                raise PathDecisionValidationError("Reverse plan TP size does not match Decode")
            if reverse_plan.remote_pcp_size != 1 or reverse_plan.remote_dcp_size != 1:
                raise PathDecisionValidationError("Reverse plan PCP/DCP sizes must be 1")
        return destination_block_ids, forward_token_start, reverse_plan

    def _activate_received_decision(
        self,
        decision: PathDecision,
        metadata: DualPathConnectorMetadata,
    ) -> None:
        result = decision.result
        request_id = result.request_key.decode_request_id
        state = self._decode_decision_states.get(request_id)
        if state is None:
            return
        is_attempt_refresh = False
        if not _is_open_decision_status(state.status):
            # A committed admission accepts only a greater-attempt refresh; the
            # logical Forward binding is neither replaced nor reinstalled.
            latest_attempt = self._latest_reverse_attempt_ids.get(request_id)
            if not (
                state.status is _DecodeDecisionStatus.COMMITTED
                and result.path is PathKind.DE_READ
                and result.reverse_attempt_id is not None
                and latest_attempt is not None
                and result.reverse_attempt_id > latest_attempt
            ):
                return
            is_attempt_refresh = True

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

        snapshot = self._decode_kv_snapshots[request_id]
        try:
            destination_block_ids, forward_token_start, reverse_plan = self._validate_committed_decision(
                decision, state, snapshot
            )
            binding = ForwardReceiveBinding(
                request_key=state.request_key,
                path=result.path,
                wire_request_id=get_external_request_id(request_id),
                decode_request_id=request_id,
                destination_block_ids=destination_block_ids,
                token_start=forward_token_start,
                token_end=snapshot.transfer_tokens,
            )
            if not is_attempt_refresh and result.path is PathKind.DE_READ and snapshot.store_load_spec is not None:
                assert self._kvpool_adapter is not None
                self._kvpool_adapter.commit_after_alloc(
                    state.request,
                    snapshot.allocated_blocks,
                    snapshot.store_load_spec,
                )
        except Exception as error:  # noqa: BLE001
            logger.error("DualPath Decode activation failed for request %s: %s", request_id, error)
            self._de_progress_deadlines.pop(request_id, None)
            if result.path is PathKind.DE_READ and result.reverse_attempt_id is not None:
                # The claim was won but no worker work was ever published.
                self._path_decision_coordinator.cancel_reverse_publication(
                    ReverseAttemptKey(state.request_key, result.reverse_attempt_id)
                )
            state.status = _DecodeDecisionStatus.ACTIVATION_FAILED
            self._path_decision_coordinator.unregister(state.request_key)
            metadata.control_failures.append(
                self._build_decode_control_failure(
                    request_id,
                    snapshot,
                    DualPathControlFailureReason.ACTIVATION_FAILED,
                )
            )
            return

        if reverse_plan is not None:
            # The reverse-send completion proof is allocated at attempt
            # acceptance, before any layer can enter the sender queue, and is
            # carried unchanged on the plan to every DE worker.
            attempt_key = ReverseAttemptKey(state.request_key, result.reverse_attempt_id)
            send_job = self._job_ledger.create_job(
                JobKind.REVERSE_SEND,
                expected_worker_count=self._expected_worker_count,
                reverse_attempt_key=attempt_key,
            )
            reverse_plan = replace(reverse_plan, reverse_send_job_id=send_job.job_id)
            self._reverse_send_job_ids[attempt_key] = send_job.job_id
            metadata.reverse_plans.append(reverse_plan)
            self._path_decision_coordinator.mark_reverse_work_published(attempt_key, send_job.job_id)
        if result.path is PathKind.DE_READ and result.reverse_attempt_id is not None:
            self._latest_reverse_attempt_ids[request_id] = result.reverse_attempt_id
        if is_attempt_refresh:
            return
        metadata.forward_receive_bindings.append(binding)
        state.status = _DecodeDecisionStatus.COMMITTED
        if reverse_plan is not None:
            self._de_progress_deadlines[request_id] = time.monotonic() + self._de_progress_watchdog_s
        self._log_decision_activation(decision, state, snapshot, reverse_plan, binding)

    def _log_decision_activation(
        self,
        decision: PathDecision,
        state: DecodePathDecisionState,
        snapshot: DecodeKVSnapshot,
        reverse_plan: ReversePlan | None,
        binding: ForwardReceiveBinding,
    ) -> None:
        # Field layout is consumed by ops log parsing; keep it stable.
        result = decision.result
        if result.path is PathKind.PE_READ:
            store_coverage = "none"
            store_end = snapshot.local_tokens
            reverse_start = snapshot.local_tokens
            reverse_end = snapshot.local_tokens
        elif result.path is PathKind.DE_READ:
            store_coverage = "partial" if snapshot.store_load_spec is not None else "skipped"
            store_end = snapshot.store_tokens
            assert reverse_plan is not None
            reverse_start = reverse_plan.token_start
            reverse_end = reverse_plan.token_end
        else:
            assert_never(result.path)
        logger.info(
            "dual_path activation key=%s/%s selected_path=%s store=%s "
            "store_range=[%s,%s) reverse_range=[%s,%s) forward_range=[%s,%s)",
            state.request_key.decode_engine_instance_id,
            state.request_key.decode_request_id,
            result.path.value,
            store_coverage,
            snapshot.local_tokens,
            store_end,
            reverse_start,
            reverse_end,
            binding.token_start,
            binding.token_end,
        )

    def _request_for_failed_job(self, job: JobRecord) -> str | None:
        if job.job_kind is JobKind.REVERSE_COMPLETION:
            for request_id, attempt_key in self._waiting_reverse_attempt_ids.items():
                if attempt_key == job.reverse_attempt_key:
                    return request_id
        if job.job_kind is JobKind.REVERSE_SEND and job.reverse_attempt_key is not None:
            return job.reverse_attempt_key.request_key.decode_request_id
        return None

    def _sweep_prefill_recovery_watchdogs(self, metadata: DualPathConnectorMetadata) -> None:
        # Expiry fails the request through the control-failure path; it never
        # releases a hold and never synthesizes a safety proof.
        now = time.monotonic()
        for request_id, deadline in list(self._recovery_deadlines.items()):
            if deadline > now:
                continue
            self._recovery_deadlines.pop(request_id)
            self._close_retry_deadlines.pop(request_id, None)
            self._pending_close_futures.pop(request_id, None)
            self._pending_close_requests.pop(request_id, None)
            self._prefill_invalid_request_ids.add(request_id)
            invalid_block_ids = self._recovery_invalid_block_ids(request_id)
            if invalid_block_ids:
                metadata.control_failures.append(
                    DualPathControlFailureMetadata(
                        request_id=request_id,
                        invalid_block_ids=invalid_block_ids,
                        reason=DualPathControlFailureReason.RECOVERY_TIMEOUT,
                    )
                )
            logger.error(
                "DualPath recovery watchdog expired for request %s; holds are retained",
                request_id,
            )

    def _recovery_invalid_block_ids(self, request_id: str) -> tuple[int, ...]:
        reverse_plan = self._prefill_reverse_plans.get(request_id)
        if reverse_plan is not None:
            block_size = self.block_size[0]
            first_block = reverse_plan.token_start // block_size
            last_block = math.ceil(reverse_plan.token_end / block_size)
            return tuple(reverse_plan.destination_block_ids[0][first_block:last_block])
        return ()

    def _sweep_decode_progress_watchdogs(self, metadata: DualPathConnectorMetadata) -> None:
        # Bounds a committed DE_READ admission whose reverse-send job has not
        # closed; expiry fails the request without touching the job ledger.
        now = time.monotonic()
        for request_id, deadline in list(self._de_progress_deadlines.items()):
            if deadline > now:
                continue
            self._de_progress_deadlines.pop(request_id)
            state = self._decode_decision_states.get(request_id)
            if state is None or state.status is not _DecodeDecisionStatus.COMMITTED:
                continue
            state.status = _DecodeDecisionStatus.ACTIVATION_FAILED
            self._path_decision_coordinator.unregister(state.request_key)
            snapshot = self._decode_kv_snapshots[request_id]
            metadata.control_failures.append(
                self._build_decode_control_failure(
                    request_id,
                    snapshot,
                    DualPathControlFailureReason.RECOVERY_TIMEOUT,
                )
            )
            logger.error(
                "DualPath DE progress watchdog expired for request %s; no safety proof is synthesized",
                request_id,
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> DualPathConnectorMetadata:
        parent_metadata = super().build_connector_meta(scheduler_output)
        # Always return the DualPath subclass (possibly with empty fields) so
        # Worker-side consumers never defend against a plain parent instance.
        metadata = DualPathConnectorMetadata()
        metadata.requests = parent_metadata.requests
        metadata.send_task = parent_metadata.send_task
        if self.dual_path_cfg.role != "decode":
            self._reconcile_prefill_deliveries()
            self._reconcile_reverse_attempt_closes()
            for deferred_request_id in list(self._prefill_deferred_deliveries):
                self._deliver_prefill_decision(deferred_request_id)
            self._sweep_prefill_recovery_watchdogs(metadata)
            metadata.reverse_receive_bindings.extend(self._prefill_pending_reverse_receive_bindings.values())
            metadata.control_failures.extend(self._prefill_control_failures.values())
            self._prefill_pending_reverse_receive_bindings.clear()
            self._prefill_control_failures.clear()
            return metadata

        coordinator = self._path_decision_coordinator

        for decision in coordinator.take_received_decisions():
            self._activate_received_decision(decision, metadata)
        self._sweep_decode_progress_watchdogs(metadata)

        now = time.monotonic()
        for request_id, state in self._decode_decision_states.items():
            if not _is_open_decision_status(state.status):
                continue
            if state.deadline > now:
                continue
            state.status = _DecodeDecisionStatus.DECISION_TIMEOUT
            self._de_progress_deadlines.pop(request_id, None)
            coordinator.unregister(state.request_key)
            snapshot = self._decode_kv_snapshots[request_id]
            metadata.control_failures.append(
                self._build_decode_control_failure(
                    request_id,
                    snapshot,
                    DualPathControlFailureReason.DECISION_TIMEOUT,
                )
            )

        assert self._kvpool_adapter is not None
        store_metadata = self._kvpool_adapter.build_connector_meta(scheduler_output)
        if (
            store_metadata.requests
            or store_metadata.unfinished_request_ids
            or store_metadata.preempted_req_ids
            or store_metadata.loading_req_ids
            or store_metadata.delayed_free_req_ids
        ):
            metadata.decode_store_metadata = store_metadata

        return metadata

    def _release_scheduler_request_state(self, request: Request) -> None:
        request_id = request.request_id
        self._lookup_results.pop(request_id, None)
        self._decode_kv_snapshots.pop(request_id, None)
        state = self._decode_decision_states.pop(request_id, None)
        if state is not None:
            self._path_decision_coordinator.unregister(state.request_key)
        self._de_progress_deadlines.pop(request_id, None)
        latest_attempt = self._latest_reverse_attempt_ids.get(request_id)
        if state is not None and latest_attempt is not None:
            attempt_key = ReverseAttemptKey(state.request_key, latest_attempt)
            send_job = self._job_ledger.get(self._reverse_send_job_ids.get(attempt_key, -1))
            if send_job is None or send_job.closed:
                self._latest_reverse_attempt_ids.pop(request_id, None)
                self._reverse_send_job_ids.pop(attempt_key, None)
            if send_job is not None and send_job.closed:
                self._job_ledger.discard(send_job.job_id)
        if self.dual_path_cfg.role == "prefill":
            self._prefill_decision_metadata.pop(request_id, None)
            self._prefill_local_tokens.pop(request_id, None)
            self._prefill_path_results.pop(request_id, None)
            forward_plan = self._prefill_forward_plans.pop(request_id, None)
            self._prefill_forward_plan_epochs.pop(request_id, None)
            self._prefill_reverse_plans.pop(request_id, None)
            self._prefill_pending_reverse_receive_bindings.pop(request_id, None)
            self._prefill_control_failures.pop(request_id, None)
            if forward_plan is not None:
                self._reqs_need_send_layerwise.pop(request_id, None)
            released_key = self._prefill_request_keys.pop(request_id, None)
            if released_key is not None:
                assert self._path_decider is not None
                self._path_decider.discard(released_key)
            self._prefill_invalid_request_ids.discard(request_id)
            self._prefill_delivered_reverse_attempts.pop(request_id, None)
            self._prefill_deferred_deliveries.discard(request_id)
            self._prefill_vacuous_reverse_request_ids.discard(request_id)
            self._reconcile_prefill_deliveries()
        # Hold/job records with unreleased holds survive request cleanup; they
        # are removed only when their jobs close.
        self._waiting_reverse_attempt_ids.pop(request_id, None)

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        delay_free = self._delay_free_for_connector(request)
        self._release_scheduler_request_state(request)
        parent_delay_free, params = super().request_finished(request, block_ids)
        return delay_free or parent_delay_free, params

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        delay_free = self._delay_free_for_connector(request)
        self._release_scheduler_request_state(request)
        parent_delay_free, params = super().request_finished_all_groups(request, block_ids)
        return delay_free or parent_delay_free, params

    def _delay_free_for_connector(self, request: Request) -> bool:
        # The Forward direction never delays the free: it is the ordinary
        # Layerwise push and follows the parent's immediate-free semantics.
        # Only the Reverse direction, whose blocks are written by the peer,
        # holds a request back.
        request_id = request.request_id
        if self.dual_path_cfg.role == "decode":
            # A final Decode request with an open reverse-send job must keep
            # the engine stepping: the delayed free retains it upstream so
            # zero-token steps keep harvesting the job report.
            latest_attempt = self._latest_reverse_attempt_ids.get(request_id)
            state = self._decode_decision_states.get(request_id)
            if latest_attempt is None or state is None:
                return False
            send_job = self._job_ledger.get(
                self._reverse_send_job_ids.get(ReverseAttemptKey(state.request_key, latest_attempt), -1)
            )
            if send_job is None or send_job.closed:
                return False
            self._pending_finished_sending.add(request_id)
            return True
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

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        self._reconcile_reverse_attempt_closes()
        finished_sending_injection: set[str] = set()
        finished_recving_injection: set[str] = set()
        worker_metadata = connector_output.kv_connector_worker_meta
        if worker_metadata is not None:
            assert isinstance(worker_metadata, DualPathWorkerMetadata), (
                f"DualPath scheduler requires DualPathWorkerMetadata, got {type(worker_metadata).__name__}"
            )
            finished_sending_injection, finished_recving_injection = self._aggregate_worker_job_facts(worker_metadata)
        if finished_sending_injection:
            if connector_output.finished_sending is None:
                connector_output.finished_sending = set()
            connector_output.finished_sending.update(finished_sending_injection)
        if finished_recving_injection:
            if connector_output.finished_recving is None:
                connector_output.finished_recving = set()
            connector_output.finished_recving.update(finished_recving_injection)
        if self._pending_ordinary_release:
            if connector_output.finished_recving is None:
                connector_output.finished_recving = set()
            connector_output.finished_recving.update(self._pending_ordinary_release)
            self._pending_ordinary_release.clear()

    def _aggregate_worker_job_facts(self, worker_metadata: DualPathWorkerMetadata) -> tuple[set[str], set[str]]:
        finished_sending_injection: set[str] = set()
        finished_recving_injection: set[str] = set()
        for job_id, report_count in worker_metadata.completed_jobs.items():
            job = self._job_ledger.get(job_id)
            if job is None:
                continue
            if self._job_ledger.record_reports(job_id, report_count):
                sending, recving = self._run_job_close_action(job)
                finished_sending_injection.update(sending)
                finished_recving_injection.update(recving)
        for job_id in worker_metadata.failed_jobs:
            job = self._job_ledger.get(job_id)
            if job is None:
                continue
            if self._job_ledger.record_failure(job_id):
                logger.error(
                    "DualPath job %s (kind=%s) reported failed; affected holds %s are retained",
                    job_id,
                    job.job_kind.value,
                    job.affected_hold_ids,
                )
                failed_request_id = self._request_for_failed_job(job)
                if failed_request_id is not None:
                    # Surface the terminal failure through the existing
                    # control-failure path at the next build pass; holds are
                    # never released and no release proof is synthesized.
                    if self.dual_path_cfg.role == "decode":
                        self._de_progress_deadlines[failed_request_id] = 0.0
                    else:
                        self._recovery_deadlines[failed_request_id] = 0.0
        return finished_sending_injection, finished_recving_injection

    def _run_job_close_action(self, job: JobRecord) -> tuple[set[str], set[str]]:
        if job.job_kind is JobKind.REVERSE_COMPLETION:
            return set(), self._close_reverse_completion_job(job)
        if job.job_kind is JobKind.REVERSE_SEND:
            attempt_key = job.reverse_attempt_key
            if attempt_key is None:
                return set(), set()
            self._path_decision_coordinator.mark_reverse_send_complete(attempt_key)
            request_id = attempt_key.request_key.decode_request_id
            self._de_progress_deadlines.pop(request_id, None)
            finished_sending: set[str] = set()
            if request_id in self._pending_finished_sending:
                self._pending_finished_sending.discard(request_id)
                self._reverse_send_job_ids.pop(attempt_key, None)
                self._latest_reverse_attempt_ids.pop(request_id, None)
                self._job_ledger.discard(job.job_id)
                finished_sending.add(request_id)
            return finished_sending, set()
        return set(), set()

    def _close_reverse_completion_job(self, job: JobRecord) -> set[str]:
        # I4: only the current waiting attempt's completion may publish the
        # request id; stale-attempt jobs are absorbed without releasing holds
        # or touching the generic finished sets.
        attempt_key = job.reverse_attempt_key
        request_id = next(
            (
                waiting_request_id
                for waiting_request_id, waiting_attempt_key in self._waiting_reverse_attempt_ids.items()
                if waiting_attempt_key == attempt_key
            ),
            None,
        )
        if request_id is None:
            return set()
        del self._waiting_reverse_attempt_ids[request_id]
        self._recovery_deadlines.pop(request_id, None)
        if self._block_pool is not None:
            for hold_id in job.affected_hold_ids:
                self._hold_ledger.release(self._block_pool, hold_id)
        for hold_id in job.affected_hold_ids:
            self._hold_ledger.discard(hold_id)
        self._job_ledger.discard(job.job_id)
        return {request_id}

    def _is_reverse_send_complete(self, attempt_key: ReverseAttemptKey) -> bool:
        """``sender_complete``: the attempt's reverse-send job is closed and
        not failed, read only from the job ledger (the sole counting authority)."""
        job_id = self._reverse_send_job_ids.get(attempt_key)
        if job_id is None:
            return False
        job = self._job_ledger.get(job_id)
        return job is not None and job.closed and not job.failed

    def shutdown(self) -> None:
        """Stop DualPath work and release all owned records and clients."""
        self._accepting_decode_admission = False
        self._accepting_prefill_decisions = False
        self._path_decision_coordinator.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.metaserver_client.close()
        if self._path_decider is not None:
            for request_key in set(self._prefill_request_keys.values()):
                self._path_decider.discard(request_key)
        self._lookup_results.clear()
        self._decode_kv_snapshots.clear()
        self._decode_decision_states.clear()
        self._prefill_request_keys.clear()
        self._prefill_decision_metadata.clear()
        self._prefill_local_tokens.clear()
        self._prefill_path_results.clear()
        for request_id in self._prefill_forward_plans:
            self._reqs_need_send_layerwise.pop(request_id, None)
        self._prefill_forward_plans.clear()
        self._prefill_forward_plan_epochs.clear()
        self._prefill_reverse_plans.clear()
        self._prefill_pending_reverse_receive_bindings.clear()
        self._prefill_control_failures.clear()
        self._prefill_delivery_futures.clear()
        self._prefill_delivered_reverse_attempts.clear()
        self._prefill_deferred_deliveries.clear()
        self._prefill_vacuous_reverse_request_ids.clear()
        self._prefill_invalid_request_ids.clear()
        if self._kvpool_adapter is not None:
            self._kvpool_adapter.close()

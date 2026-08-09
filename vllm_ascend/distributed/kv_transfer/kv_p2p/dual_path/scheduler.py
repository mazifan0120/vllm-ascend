# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side admission and path-decision lifecycle for DualPath."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple, assert_never

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.kvpool_adapter import (
    KVPoolSchedulerAdapter,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    BlockIdGroups,
    DualPathConnectorMetadata,
    DualPathControlFailureMetadata,
    DualPathControlFailureReason,
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
    RoundRobinPathPolicy,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DUAL_PATH_PROTOCOL_VERSION,
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
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
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
    TIMED_OUT = "TIMED_OUT"
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


def _classify_store_hit(
    spec: LoadSpec | None,
    local_tokens: int,
    ready_tokens: int,
    transfer_tokens: int,
) -> tuple[bool, int]:
    """A Store hit is full only when the detached spec covers exactly up to the
    ready boundary; external tokens then stop there instead of the transfer target."""
    store_full = (
        spec is not None and spec.vllm_cached_tokens == local_tokens and spec.kvpool_cached_tokens == ready_tokens
    )
    return store_full, (ready_tokens if store_full else transfer_tokens) - local_tokens


def _expected_prefill_token_end(decision_request: PathDecisionRequest, need_truncate: bool) -> int:
    """Hybrid (truncating) deployments advertise one fewer token than the prompt."""
    return decision_request.target_tokens if need_truncate else decision_request.target_tokens + 1


def _is_open_decision_status(status: _DecodeDecisionStatus) -> bool:
    """Only PENDING decisions still accept an outcome or a timeout."""
    if status is _DecodeDecisionStatus.PENDING:
        return True
    if status in (
        _DecodeDecisionStatus.COMMITTED,
        _DecodeDecisionStatus.TIMED_OUT,
        _DecodeDecisionStatus.ACTIVATION_FAILED,
    ):
        return False
    assert_never(status)


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
        self._accepting_pe_decisions = True
        # PE fields exist on both roles; Decode keeps a None decider and empty
        # maps rather than constructing policy state it never owns.
        self._path_decider: PathDecisionDecider | None = None
        self._pe_request_keys: dict[str, DualPathRequestKey] = {}
        self._pe_decision_metadata: dict[str, DualPathDecisionMetadata] = {}
        self._pe_prefill_local_tokens: dict[str, int] = {}
        self._pe_path_results: dict[str, PathDecisionResult] = {}
        self._pe_forward_plans: dict[str, ForwardPlan] = {}
        self._pe_forward_send_infos: dict[str, SendReqInfo] = {}
        self._pe_pending_reverse_receive_bindings: dict[str, ReverseReceiveBinding] = {}
        self._pe_control_failures: dict[str, DualPathControlFailureMetadata] = {}
        self._pe_delivery_futures: dict[DualPathRequestKey, Future[None]] = {}
        self._pe_invalid_request_ids: set[str] = set()
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
        self._pe_control_failures[request_id] = failure

    def _sweep_pe_delivery(self, released_key: DualPathRequestKey | None = None) -> None:
        if self._path_decider is None:
            return

        active_keys = frozenset(self._pe_request_keys.values())
        sweep_keys = set(self._pe_delivery_futures)
        if released_key is not None:
            sweep_keys.add(released_key)

        for request_key in sweep_keys:
            delivery_future = self._pe_delivery_futures.get(request_key)
            if delivery_future is not None and delivery_future.done():
                delivery_failed = delivery_future.cancelled() or delivery_future.exception() is not None
                if delivery_failed:
                    request_id = next(
                        (
                            request_id
                            for request_id, retained_key in self._pe_request_keys.items()
                            if retained_key == request_key
                        ),
                        None,
                    )
                    result = self._pe_path_results.get(request_id) if request_id is not None else None
                    if result is not None and request_id not in self._pe_invalid_request_ids:
                        if result.path is PathKind.DE_READ:
                            binding = self._pe_pending_reverse_receive_bindings.get(request_id)
                            if binding is not None:
                                destination_block_ids = binding.destination_block_ids[0]
                                token_start = binding.token_start
                                token_end = binding.token_end
                            else:
                                forward_plan = self._pe_forward_plans.get(request_id)
                                token_start = self._pe_prefill_local_tokens.get(request_id)
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
                            self._pe_invalid_request_ids.add(request_id)
                            self._pe_pending_reverse_receive_bindings.pop(request_id, None)
                        elif result.path is PathKind.PE_READ:
                            pass
                        else:
                            assert_never(result.path)
            if request_key in active_keys:
                continue
            if delivery_future is not None and not delivery_future.done():
                continue
            self._path_decider.discard(request_key)
            self._pe_delivery_futures.pop(request_key, None)

    def _decide_prefill_path_for_admission(
        self,
        request: Request,
        parent_result: tuple[int, bool],
        prefill_local_tokens: int,
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if (
            not self._accepting_pe_decisions
            or params is None
            or params.get("do_remote_decode") is not True
            or "dual_path" not in params
        ):
            return parent_result
        self._sweep_pe_delivery()

        request_id = request.request_id
        if request_id in self._pe_invalid_request_ids:
            return parent_result

        try:
            metadata = DualPathDecisionMetadata.from_dict(params["dual_path"])
        except PathDecisionValidationError as error:
            logger.error(
                "DualPath Prefill decision metadata is invalid for request %s: %s",
                request_id,
                error,
            )
            self._pe_invalid_request_ids.add(request_id)
            return parent_result

        if metadata.protocol_version != DUAL_PATH_PROTOCOL_VERSION:
            logger.error(
                "DualPath Prefill decision protocol version mismatch for request %s: expected %s, got %s",
                request_id,
                DUAL_PATH_PROTOCOL_VERSION,
                metadata.protocol_version,
            )
            self._pe_invalid_request_ids.add(request_id)
            return parent_result

        decision_request = metadata.decision_request
        effective_prefill_tokens = _expected_prefill_token_end(decision_request, self.need_truncate)
        if not 0 <= prefill_local_tokens <= effective_prefill_tokens:
            logger.error(
                "DualPath Prefill local tokens are invalid for request %s: expected 0 <= %s <= %s",
                request_id,
                prefill_local_tokens,
                effective_prefill_tokens,
            )
            return parent_result

        request_key = decision_request.request_key
        self._pe_request_keys[request_id] = request_key
        self._pe_decision_metadata[request_id] = metadata
        self._pe_prefill_local_tokens.setdefault(request_id, prefill_local_tokens)
        assert self._path_decider is not None
        try:
            result = self._path_decider.decide(decision_request, prefill_local_tokens)
        except Exception as error:  # noqa: BLE001
            logger.error(
                "DualPath Prefill decision failed locally for request %s: %s",
                request_id,
                error,
            )
            self._pe_invalid_request_ids.add(request_id)
            return parent_result

        self._pe_path_results.setdefault(request_id, result)

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
        metadata = self._pe_decision_metadata[request.request_id]
        decision_request = metadata.decision_request
        result = self._pe_path_results[request.request_id]
        if result.request_key != decision_request.request_key:
            raise PathDecisionValidationError("retained result key does not match the Decision Request key")

        token_end = request.num_prompt_tokens
        expected_token_end = _expected_prefill_token_end(decision_request, self.need_truncate)
        if token_end != expected_token_end:
            raise PathDecisionValidationError(
                f"effective Prefill token count {token_end} does not match the expected transfer target "
                f"{expected_token_end}"
            )
        if not 0 <= token_start < token_end:
            raise PathDecisionValidationError("Forward token range must satisfy 0 <= token_start < token_end")

        topology_fields = ("remote_tp_size", "remote_pcp_size", "remote_dcp_size")
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
        # get_block_ids() may re-enter and mutate kv_transfer_params; re-read the advertised table.
        advertised_destination = tuple(tuple(group) for group in params["remote_block_ids"])
        if advertised_destination != destination_block_ids:
            raise PathDecisionValidationError("destination block table changed during Forward plan installation")
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

    def _try_install_forward_plan(self, request: Request, blocks: KVCacheBlocks) -> None:
        request_id = request.request_id
        result = self._pe_path_results.get(request_id)
        if result is None:
            return
        token_start = self._pe_decision_metadata[request_id].decision_request.decode_local_tokens

        try:
            prepared = self._prepare_forward_plan(request, blocks, token_start)
        except (KeyError, PathDecisionValidationError, TypeError) as error:
            logger.error(
                "DualPath Prefill Forward plan is invalid for request %s: %s",
                request_id,
                error,
            )
            self._pe_invalid_request_ids.add(request_id)
            self._pe_path_results.pop(request_id, None)
            return
        if prepared is None:
            return
        plan, send_req_info = prepared

        existing_plan = self._pe_forward_plans.get(request_id)
        if existing_plan is not None:
            if existing_plan == plan:
                return
            raise RuntimeError(
                f"DualPath Prefill request {request_id} got a conflicting duplicate Forward plan; "
                "the original plan is preserved"
            )

        self._pe_forward_plans[request_id] = plan
        self._pe_forward_send_infos[request_id] = send_req_info
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
        metadata = self._pe_decision_metadata[request_id]
        decision_request = metadata.decision_request
        if result.request_key != decision_request.request_key:
            raise PathDecisionValidationError("retained result key does not match the Decision Request key")

        token_start = self._pe_prefill_local_tokens[request_id]
        token_split = decision_request.decode_store_tokens
        ready_tokens = decision_request.target_tokens
        token_end = request.num_prompt_tokens
        if not token_start < token_split < ready_tokens <= token_end:
            raise PathDecisionValidationError("DE_READ token ranges must satisfy L_PE < K_DE < R <= T")
        if not 0 <= decision_request.decode_local_tokens <= token_split:
            raise PathDecisionValidationError("Decode local tokens must not exceed the DE_READ split")

        prepared = self._prepare_forward_plan(request, blocks, token_split)
        if prepared is None:
            return None
        forward_plan, send_req_info = prepared
        wire_request_id = get_external_request_id(request_id)
        binding = ReverseReceiveBinding(
            request_key=result.request_key,
            wire_request_id=wire_request_id,
            prefill_request_id=request_id,
            destination_block_ids=forward_plan.source_block_ids,
            token_start=token_start,
            token_end=token_split,
        )
        parallel_config = self.vllm_config.parallel_config
        reverse_plan = ReversePlan(
            request_key=result.request_key,
            wire_request_id=wire_request_id,
            token_start=token_start,
            token_end=token_split,
            source_block_ids=forward_plan.destination_block_ids,
            destination_block_ids=forward_plan.source_block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_block_sizes=tuple(self.block_size),
            remote_tp_size=parallel_config.tensor_parallel_size,
            remote_pcp_size=parallel_config.prefill_context_parallel_size,
            remote_dcp_size=parallel_config.decode_context_parallel_size,
        )

        existing_plan = self._pe_forward_plans.get(request_id)
        if existing_plan is not None and existing_plan != forward_plan:
            raise RuntimeError(
                f"DualPath Prefill request {request_id} got a conflicting duplicate Forward plan; "
                "the original plan is preserved"
            )
        existing_binding = self._pe_pending_reverse_receive_bindings.get(request_id)
        conflicting_binding = any(
            retained != binding
            and (retained.request_key == binding.request_key or retained.wire_request_id == binding.wire_request_id)
            for retained in self._pe_pending_reverse_receive_bindings.values()
        )
        if (existing_binding is not None and existing_binding != binding) or conflicting_binding:
            raise RuntimeError(
                f"DualPath Prefill request {request_id} got a conflicting duplicate Reverse receive binding; "
                "the original binding is preserved"
            )
        if existing_plan is not None and result.request_key in self._pe_delivery_futures:
            return reverse_plan

        self._pe_pending_reverse_receive_bindings[request_id] = binding
        self._pe_forward_plans[request_id] = forward_plan
        self._pe_forward_send_infos[request_id] = send_req_info
        self._reqs_need_send_layerwise[request_id] = send_req_info
        return reverse_plan

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        if not self._is_dual_path_decode_admission(request):
            parent_result = super().get_num_new_matched_tokens(request, num_computed_tokens)
            if self.dual_path_cfg.role == "prefill":
                return self._decide_prefill_path_for_admission(request, parent_result, num_computed_tokens)
            return parent_result

        request_id = request.request_id
        transfer_tokens = self._hybrid_prefill_token_count(request.num_tokens)
        ready_tokens = _decode_ready_token_count(request.num_tokens)
        local_tokens = num_computed_tokens
        if local_tokens < 0 or ready_tokens > transfer_tokens:
            logger.warning(
                "DualPath initial admission is inconsistent for request %s; using local allocation: "
                "local_tokens=%d, ready_tokens=%d, transfer_tokens=%d",
                request_id,
                local_tokens,
                ready_tokens,
                transfer_tokens,
            )
            return 0, False
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
            cached_store_full, expected_external_tokens = _classify_store_hit(
                cached_spec, local_tokens, ready_tokens, transfer_tokens
            )
            if (
                cached_spec_is_current
                and cached.local_tokens == local_tokens
                and cached.external_tokens == expected_external_tokens
            ):
                # Identical duplicate lookup (e.g. allocation-failure retry):
                # reuse the detached result instead of re-probing the KV pool.
                return cached.external_tokens, True
            # Changed admission facts invalidate the unbound result; discard it before
            # the fresh lookup so a failing re-probe cannot leave it behind.
            del self._lookup_results[request_id]

        assert self._kvpool_adapter is not None
        detached_spec = self._kvpool_adapter.lookup(request, local_tokens)
        store_full, external_tokens = _classify_store_hit(detached_spec, local_tokens, ready_tokens, transfer_tokens)
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
        if request_id in self._pe_invalid_request_ids:
            return
        result = self._pe_path_results.get(request_id)
        if result is None:
            return

        reverse_plan: ReversePlan | None = None
        try:
            if result.path is PathKind.DE_READ:
                reverse_plan = self._activate_de_read_path(request, blocks, result)
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

        result = self._pe_path_results.get(request_id)
        if result is None or request_id not in self._pe_forward_plans:
            return

        request_key = result.request_key
        if request_key in self._pe_delivery_futures:
            return
        metadata = self._pe_decision_metadata[request_id]
        decision = PathDecision(
            protocol_version=DUAL_PATH_PROTOCOL_VERSION,
            result=result,
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
            self._invalidate_prefill_activation(request, blocks, result, discard_installed_plans=True)
            return
        self._pe_delivery_futures[request_key] = delivery_future

        def log_delivery_failure(completed_future: Future[None]) -> None:
            if completed_future.cancelled():
                logger.warning(
                    "dual_path delivery key=%s/%s protocol=%s delivery_terminal=CANCELLED",
                    request_key.decode_engine_instance_id,
                    request_key.decode_request_id,
                    DUAL_PATH_PROTOCOL_VERSION,
                )
                return
            error = completed_future.exception()
            if error is not None:
                logger.error(
                    "dual_path delivery key=%s/%s protocol=%s delivery_terminal=FAILED "
                    "failure_source=DELIVERY error=%s",
                    request_key.decode_engine_instance_id,
                    request_key.decode_request_id,
                    DUAL_PATH_PROTOCOL_VERSION,
                    error,
                )
                return
            logger.info(
                "dual_path delivery key=%s/%s protocol=%s delivery_terminal=SUCCEEDED",
                request_key.decode_engine_instance_id,
                request_key.decode_request_id,
                DUAL_PATH_PROTOCOL_VERSION,
            )

        delivery_future.add_done_callback(log_delivery_failure)

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
            decision_request = self._pe_decision_metadata[request_id].decision_request
            self._stage_prefill_activation_failure(
                request_id,
                blocks.get_block_ids()[0],
                self._pe_prefill_local_tokens[request_id],
                decision_request.decode_store_tokens,
            )
        self._pe_invalid_request_ids.add(request_id)
        self._pe_path_results.pop(request_id, None)
        if not discard_installed_plans:
            return
        self._pe_pending_reverse_receive_bindings.pop(request_id, None)
        self._pe_forward_plans.pop(request_id, None)
        owned_send_req_info = self._pe_forward_send_infos.pop(request_id, None)
        if owned_send_req_info is not None and self._reqs_need_send_layerwise.get(request_id) is owned_send_req_info:
            self._reqs_need_send_layerwise.pop(request_id)

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
                "the original admission is preserved"
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
        store_full, expected_external_tokens = _classify_store_hit(
            detached_spec, local_tokens, ready_tokens, transfer_tokens
        )
        if not num_external_tokens == cached_external_tokens == expected_external_tokens:
            raise RuntimeError(
                f"DualPath request {request_id} external-token mismatch: vLLM allocated "
                f"{num_external_tokens}, cached lookup expected {cached_external_tokens}, "
                f"and transfer accounting expected {expected_external_tokens}"
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
        if store_full:
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
        existing_store_full, _ = _classify_store_hit(
            existing.store_load_spec, existing.local_tokens, ready_tokens, transfer_tokens
        )
        duplicate_local_tokens = (ready_tokens if existing_store_full else transfer_tokens) - num_external_tokens
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
            protocol_version=DUAL_PATH_PROTOCOL_VERSION,
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
        request_id = result.request_key.decode_request_id
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
            if reverse_plan.wire_request_id != get_external_request_id(request_id):
                raise PathDecisionValidationError("Reverse plan wire id does not match the Decode request")
            if reverse_plan.token_end != snapshot.store_tokens:
                raise PathDecisionValidationError("Reverse plan range does not end at the frozen Store boundary")
            if reverse_plan.source_block_ids != destination_block_ids:
                raise PathDecisionValidationError("Reverse plan source does not match the advertised Decode table")
            if len(reverse_plan.remote_block_sizes) != len(self.block_size):
                raise PathDecisionValidationError("Reverse plan block-size group count does not match Decode")
            frozen_wrapper_blocks = tuple(tuple(group) for group in snapshot.allocated_blocks.get_block_ids())
            if frozen_wrapper_blocks != snapshot.final_block_ids:
                raise PathDecisionValidationError("retained allocation wrapper no longer matches the snapshot")
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
        if not _is_open_decision_status(state.status):
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
            if result.path is PathKind.DE_READ and snapshot.store_load_spec is not None:
                assert self._kvpool_adapter is not None
                self._kvpool_adapter.commit_after_alloc(
                    state.request,
                    snapshot.allocated_blocks,
                    snapshot.store_load_spec,
                )
        except Exception as error:  # noqa: BLE001
            logger.error("DualPath Decode activation failed for request %s: %s", request_id, error)
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
            metadata.reverse_plans.append(reverse_plan)
        metadata.forward_receive_bindings.append(binding)
        state.status = _DecodeDecisionStatus.COMMITTED
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
            "dual_path activation key=%s/%s protocol=%s selected_path=%s store=%s "
            "store_range=[%s,%s) reverse_range=[%s,%s) forward_range=[%s,%s)",
            state.request_key.decode_engine_instance_id,
            state.request_key.decode_request_id,
            decision.protocol_version,
            result.path.value,
            store_coverage,
            snapshot.local_tokens,
            store_end,
            reverse_start,
            reverse_end,
            binding.token_start,
            binding.token_end,
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
            self._sweep_pe_delivery()
            metadata.reverse_receive_bindings.extend(self._pe_pending_reverse_receive_bindings.values())
            metadata.control_failures.extend(self._pe_control_failures.values())
            self._pe_pending_reverse_receive_bindings.clear()
            self._pe_control_failures.clear()
            return metadata

        coordinator = self._path_decision_coordinator

        for decision in coordinator.take_received_decisions():
            self._activate_received_decision(decision, metadata)

        now = time.monotonic()
        for request_id, state in self._decode_decision_states.items():
            if not _is_open_decision_status(state.status):
                continue
            if state.deadline > now:
                continue
            state.status = _DecodeDecisionStatus.TIMED_OUT
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
        if self.dual_path_cfg.role == "prefill":
            self._pe_decision_metadata.pop(request_id, None)
            self._pe_prefill_local_tokens.pop(request_id, None)
            self._pe_path_results.pop(request_id, None)
            self._pe_forward_plans.pop(request_id, None)
            self._pe_pending_reverse_receive_bindings.pop(request_id, None)
            self._pe_control_failures.pop(request_id, None)
            owned_send_req_info = self._pe_forward_send_infos.pop(request_id, None)
            send_req_info = self._reqs_need_send_layerwise.get(request_id)
            if send_req_info is not None and send_req_info is owned_send_req_info and send_req_info.request is request:
                self._reqs_need_send_layerwise.pop(request_id)
            released_key = self._pe_request_keys.pop(request_id, None)
            self._pe_invalid_request_ids.discard(request_id)
            self._sweep_pe_delivery(released_key)

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        self._release_scheduler_request_state(request)
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._release_scheduler_request_state(request)
        return super().request_finished_all_groups(request, block_ids)

    def shutdown(self) -> None:
        """Stop DualPath work and release all owned records and clients."""
        self._accepting_decode_admission = False
        self._accepting_pe_decisions = False
        self._path_decision_coordinator.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.metaserver_client.close()
        if self._path_decider is not None:
            retained_keys = set(self._pe_request_keys.values())
            retained_keys.update(self._pe_delivery_futures)
            for request_key in retained_keys:
                self._path_decider.discard(request_key)
        self._lookup_results.clear()
        self._decode_kv_snapshots.clear()
        self._decode_decision_states.clear()
        self._pe_request_keys.clear()
        self._pe_decision_metadata.clear()
        self._pe_prefill_local_tokens.clear()
        self._pe_path_results.clear()
        for request_id, owned_send_req_info in self._pe_forward_send_infos.items():
            if self._reqs_need_send_layerwise.get(request_id) is owned_send_req_info:
                self._reqs_need_send_layerwise.pop(request_id)
        self._pe_forward_send_infos.clear()
        self._pe_forward_plans.clear()
        self._pe_pending_reverse_receive_bindings.clear()
        self._pe_control_failures.clear()
        self._pe_delivery_futures.clear()
        self._pe_invalid_request_ids.clear()
        if self._kvpool_adapter is not None:
            self._kvpool_adapter.close()

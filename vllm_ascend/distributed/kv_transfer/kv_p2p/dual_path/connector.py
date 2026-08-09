# SPDX-License-Identifier: Apache-2.0
"""Dual-path KV transfer built on ``MooncakeLayerwiseConnector``.

``DualPathConnector`` preserves ordinary Layerwise behavior while adding
Decode admission through the local Store, Prefill-owned path selection, and
production ``PE_READ`` and ``DE_READ`` transfer routes. Store-full requests
remain local to Decode; non-full requests use the selected forward or split
Reverse/Forward route.

Why inheritance + a custom ``__init__``:
    ``MooncakeLayerwiseConnector.__init__`` hard-instantiates its own
    ``MooncakeLayerwiseConnectorScheduler`` / ``MooncakeLayerwiseConnectorWorker``
    (see ``mooncake_layerwise_connector.py``). To give DualPath its own
    scheduler/worker subclasses, we cannot call ``super().__init__``; instead we
    replicate the parent's facade setup and build OUR subclasses, calling
    ``KVConnectorBase_V1.__init__`` directly. The copied facade state is exactly
    ``_is_kv_producer``, ``engine_id``, ``_connector_metadata``,
    ``connector_scheduler``, and ``connector_worker``; a drift guard in the
    unit tests fails if the parent facade grows additional state.

Block forwarding is free:
    Because ``DualPathConnector`` IS-A ``MooncakeLayerwiseConnector``,
    ``AscendMultiConnector.update_state_after_alloc`` already forwards it the
    real blocks whenever it wins the first-wins routing (its
    ``isinstance(c, MooncakeLayerwiseConnector)`` check covers subclasses). No
    change to ``AscendMultiConnector`` is required.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, assert_never

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.kvpool_adapter import (
    KVPoolAdapter,
    KVPoolWorkerAdapter,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
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
    Path,
    PathDecisionDecider,
    PathDecisionRequest,
    PathDecisionResult,
    PathDecisionValidationError,
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
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
    ReqMeta,
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

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
        AscendConnectorMetadata,
    )


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


class DecodeDecisionStatus(str, Enum):
    PENDING = "PENDING"
    COMMITTED = "COMMITTED"
    TIMED_OUT = "TIMED_OUT"
    ACTIVATION_FAILED = "ACTIVATION_FAILED"


class _SplitPhase(str, Enum):
    SKIPPED = "SKIPPED"
    PENDING = "PENDING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass(slots=True)
class _SplitTracker:
    """Mutable DE-local execution state for one split request."""

    store_phase: _SplitPhase
    reverse_phase: _SplitPhase
    forward_phase: _SplitPhase
    store_destination_slice: tuple[int, ...]
    forward_destination_slice: tuple[int, ...]
    plan: ReversePlan | None
    reverse_submitted: bool
    terminal_published: bool


@dataclass
class DecodePathDecisionState:
    request_key: DualPathRequestKey
    decision_request: PathDecisionRequest
    request: Request
    deadline: float
    status: DecodeDecisionStatus


def build_remote_decode_message(
    scheduler: DualPathConnectorScheduler,
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
        "remote_block_size": scheduler.block_size,
        "remote_engine_id": scheduler.engine_id,
        "remote_host": scheduler.side_channel_host,
        "remote_port": scheduler.side_channel_port,
        "remote_tp_size": scheduler.vllm_config.parallel_config.tensor_parallel_size,
        "remote_pcp_size": scheduler.vllm_config.parallel_config.prefill_context_parallel_size,
        "remote_dcp_size": scheduler.vllm_config.parallel_config.decode_context_parallel_size,
        "remote_cached_tokens": snapshot.local_tokens,
        "dual_path": decision_metadata.to_dict(),
    }


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
        self._lookup_results: dict[str, tuple[int, int, LoadSpec | None]] = {}
        self._decode_kv_snapshots: dict[str, DecodeKVSnapshot] = {}
        self._decode_decision_states: dict[str, DecodePathDecisionState] = {}
        self._decision_timeout_seconds: int | None = None
        self._kvpool_adapter: KVPoolAdapter | None = None
        self._accepting_decode_admission = True
        self._accepting_pe_decisions = True
        # PE fields exist on both roles; Decode keeps a None decider and empty
        # maps rather than constructing policy state it never owns.
        self._path_decider: PathDecisionDecider | None = None
        self._pe_request_keys: dict[str, DualPathRequestKey] = {}
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
            if (
                isinstance(decision_timeout_seconds, bool)
                or not isinstance(decision_timeout_seconds, int)
                or decision_timeout_seconds <= 0
            ):
                raise ValueError("VLLM_ASCEND_DUALPATH_DECISION_TIMEOUT must be an integer greater than zero")
            self._decision_timeout_seconds = decision_timeout_seconds
            self._kvpool_adapter = KVPoolAdapter(vllm_config, kv_cache_config)
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
        existing = self._pe_control_failures.get(request_id)
        if existing is not None and existing != failure:
            raise RuntimeError(
                f"DualPath Prefill request {request_id} got a conflicting local control failure; "
                "the original failure is preserved"
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
                        if result.path is Path.DE_READ:
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
                        elif result.path is Path.PE_READ:
                            pass
                        else:
                            assert_never(result.path)
            if request_key in active_keys:
                continue
            if delivery_future is not None and not delivery_future.done():
                continue
            self._path_decider.discard(request_key)
            self._pe_delivery_futures.pop(request_key, None)

    def _handle_prefill_decision(
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
        effective_prefill_tokens = (
            decision_request.target_tokens if self.need_truncate else decision_request.target_tokens + 1
        )
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

        retained_result = self._pe_path_results.get(request_id)
        if retained_result is None:
            self._pe_path_results[request_id] = result
        elif retained_result != result:
            self._pe_invalid_request_ids.add(request_id)
            raise RuntimeError(
                f"DualPath Prefill request {request_id} got a conflicting retained path result; "
                "the original result is preserved"
            )

        eligibility = "singleton" if prefill_local_tokens >= decision_request.decode_store_tokens else "policy"
        if result.path is Path.PE_READ:
            store_coverage = "none"
        elif result.path is Path.DE_READ:
            store_coverage = (
                "partial" if decision_request.decode_store_tokens > decision_request.decode_local_tokens else "skipped"
            )
        else:
            assert_never(result.path)
        logger.info(
            "dual_path decision key=%s/%s L_DE=%s K_DE=%s L_PE=%s R=%s T=%s eligibility=%s selected_path=%s store=%s",
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

        if result.path is Path.PE_READ:
            return 0, False
        elif result.path is Path.DE_READ:
            reverse_tokens = decision_request.decode_store_tokens - self._pe_prefill_local_tokens[request_id]
            assert reverse_tokens > 0
            return reverse_tokens, True
        else:
            assert_never(result.path)

    def _prepare_forward_plan(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        token_start: int,
    ) -> tuple[ForwardPlan, SendReqInfo] | None:
        params = request.kv_transfer_params
        assert params is not None
        metadata = DualPathDecisionMetadata.from_dict(params["dual_path"])
        if metadata.protocol_version != DUAL_PATH_PROTOCOL_VERSION:
            raise PathDecisionValidationError(
                f"protocol version must be {DUAL_PATH_PROTOCOL_VERSION}, got {metadata.protocol_version}"
            )
        decision_request = metadata.decision_request
        result = self._pe_path_results[request.request_id]
        if result.request_key != decision_request.request_key:
            raise PathDecisionValidationError("retained result key does not match the Decision Request key")

        token_end = request.num_prompt_tokens
        expected_token_end = (
            decision_request.target_tokens if self.need_truncate else decision_request.target_tokens + 1
        )
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
        missing_fields = [field for field in required_fields if field not in params]
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
            if not group or any(isinstance(block_id, bool) or not isinstance(block_id, int) for block_id in group):
                raise PathDecisionValidationError("destination block groups must contain integer block ids")
            if len(group) * block_size < token_end:
                raise PathDecisionValidationError("destination block table does not cover token_end")

        source_block_ids = tuple(tuple(group) for group in blocks.get_block_ids())
        if len(source_block_ids) != len(local_block_sizes):
            raise PathDecisionValidationError("source block-table group count does not match block sizes")
        for group in source_block_ids:
            if any(isinstance(block_id, bool) or not isinstance(block_id, int) for block_id in group):
                raise PathDecisionValidationError("source block groups must contain integer block ids")
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
        token_start = DualPathDecisionMetadata.from_dict(
            request.kv_transfer_params["dual_path"]
        ).decision_request.decode_local_tokens

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
        metadata = DualPathDecisionMetadata.from_dict(params["dual_path"])
        if metadata.protocol_version != DUAL_PATH_PROTOCOL_VERSION:
            raise PathDecisionValidationError(
                f"protocol version must be {DUAL_PATH_PROTOCOL_VERSION}, got {metadata.protocol_version}"
            )
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

    def _emit_prefill_control_failure(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        token_start: int,
        token_end: int,
    ) -> None:
        destination_block_ids = blocks.get_block_ids()[0]
        self._stage_prefill_activation_failure(
            request.request_id,
            destination_block_ids,
            token_start,
            token_end,
        )

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        if not self._is_dual_path_decode_admission(request):
            parent_result = super().get_num_new_matched_tokens(request, num_computed_tokens)
            if self.dual_path_cfg.role == "prefill" and request.kv_transfer_params is not None:
                return self._handle_prefill_decision(request, parent_result, num_computed_tokens)
            return parent_result

        request_id = request.request_id
        if request_id in self._decode_kv_snapshots:
            raise RuntimeError(
                f"DualPath request {request_id} is already admitted; a new initial lookup is a lifecycle error"
            )

        transfer_tokens = self._hybrid_prefill_token_count(request.num_tokens)
        ready_tokens = max(request.num_tokens - 1, 0)
        local_tokens = num_computed_tokens
        if local_tokens < 0 or ready_tokens > transfer_tokens:
            raise RuntimeError(
                f"DualPath request {request_id} initial admission requires "
                f"0 <= local_tokens ({local_tokens}) and ready_tokens ({ready_tokens}) "
                f"<= transfer_tokens ({transfer_tokens})"
            )
        if local_tokens >= ready_tokens:
            # HBM-complete: no KVPool lookup, no Task-01 state.
            self._lookup_results.pop(request_id, None)
            return 0, False

        if not 0 <= local_tokens < ready_tokens <= transfer_tokens:
            raise RuntimeError(
                f"DualPath request {request_id} Store lookup requires "
                f"0 <= local_tokens ({local_tokens}) < ready_tokens ({ready_tokens}) "
                f"<= transfer_tokens ({transfer_tokens})"
            )

        cached = self._lookup_results.get(request_id)
        if cached is not None:
            cached_spec = cached[2]
            cached_spec_is_current = cached_spec is None or (
                cached_spec.vllm_cached_tokens == local_tokens and cached_spec.kvpool_cached_tokens <= ready_tokens
            )
            cached_store_full = (
                cached_spec is not None
                and cached_spec.vllm_cached_tokens == local_tokens
                and cached_spec.kvpool_cached_tokens == ready_tokens
            )
            expected_external_tokens = (ready_tokens if cached_store_full else transfer_tokens) - local_tokens
            if cached_spec_is_current and cached[0] == local_tokens and cached[1] == expected_external_tokens:
                # Identical duplicate lookup (e.g. allocation-failure retry):
                # reuse the detached result instead of re-probing the KV pool.
                return cached[1], True
            # Changed admission facts invalidate the unbound result; discard it before
            # the fresh lookup so a failing re-probe cannot leave it behind.
            del self._lookup_results[request_id]

        assert self._kvpool_adapter is not None
        detached_spec = self._kvpool_adapter.lookup(request, local_tokens)
        store_full = (
            detached_spec is not None
            and detached_spec.vllm_cached_tokens == local_tokens
            and detached_spec.kvpool_cached_tokens == ready_tokens
        )
        external_tokens = (ready_tokens if store_full else transfer_tokens) - local_tokens
        self._lookup_results[request_id] = (local_tokens, external_tokens, detached_spec)
        return external_tokens, True

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        params = request.kv_transfer_params
        if self.dual_path_cfg.role == "prefill" and params is not None and "dual_path" in params:
            request_id = request.request_id
            if request_id in self._pe_invalid_request_ids:
                return
            result = self._pe_path_results.get(request_id)
            if result is None:
                return

            reverse_plan: ReversePlan | None = None
            try:
                if result.path is Path.DE_READ:
                    reverse_plan = self._activate_de_read_path(request, blocks, result)
                elif result.path is Path.PE_READ:
                    self._try_install_forward_plan(request, blocks)
                else:
                    assert_never(result.path)
            except (KeyError, PathDecisionValidationError, RuntimeError, TypeError) as error:
                logger.error(
                    "DualPath Prefill activation failed locally for request %s: %s",
                    request_id,
                    error,
                )
                if result.path is Path.DE_READ:
                    decision_request = DualPathDecisionMetadata.from_dict(params["dual_path"]).decision_request
                    self._emit_prefill_control_failure(
                        request,
                        blocks,
                        self._pe_prefill_local_tokens[request_id],
                        decision_request.decode_store_tokens,
                    )
                self._pe_invalid_request_ids.add(request_id)
                self._pe_path_results.pop(request_id, None)
                return

            result = self._pe_path_results.get(request_id)
            if result is None or request_id not in self._pe_forward_plans:
                return

            request_key = result.request_key
            if request_key in self._pe_delivery_futures:
                return
            metadata = DualPathDecisionMetadata.from_dict(params["dual_path"])
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
                if result.path is Path.DE_READ:
                    decision_request = metadata.decision_request
                    self._emit_prefill_control_failure(
                        request,
                        blocks,
                        self._pe_prefill_local_tokens[request_id],
                        decision_request.decode_store_tokens,
                    )
                self._pe_invalid_request_ids.add(request_id)
                self._pe_path_results.pop(request_id, None)
                self._pe_pending_reverse_receive_bindings.pop(request_id, None)
                self._pe_forward_plans.pop(request_id, None)
                owned_send_req_info = self._pe_forward_send_infos.pop(request_id, None)
                if (
                    owned_send_req_info is not None
                    and self._reqs_need_send_layerwise.get(request_id) is owned_send_req_info
                ):
                    self._reqs_need_send_layerwise.pop(request_id)
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
            return

        if not self._is_dual_path_decode_admission(request):
            # Non-selected requests delegate untouched; in particular a request
            # resumed after a completed async load (its admission flag is already
            # consumed and num_external_tokens == 0) must not re-enter the
            # duplicate-bind check against its own earlier snapshot.
            return super().update_state_after_alloc(request, blocks, num_external_tokens)

        request_id = request.request_id
        allocated_block_ids = blocks.get_block_ids()
        frozen_block_ids = tuple(tuple(group) for group in allocated_block_ids)

        existing = self._decode_kv_snapshots.get(request_id)
        if existing is not None:
            ready_tokens = max(request.num_tokens - 1, 0)
            transfer_tokens = self._hybrid_prefill_token_count(request.num_tokens)
            existing_store_full = (
                existing.store_load_spec is not None
                and existing.store_load_spec.vllm_cached_tokens == existing.local_tokens
                and existing.store_load_spec.kvpool_cached_tokens == ready_tokens
            )
            duplicate_local_tokens = (ready_tokens if existing_store_full else transfer_tokens) - num_external_tokens
            if (
                existing.transfer_tokens == transfer_tokens
                and existing.local_tokens == duplicate_local_tokens
                and existing.external_tokens == num_external_tokens
                and existing.final_block_ids == frozen_block_ids
            ):
                return
            raise RuntimeError(
                f"DualPath request {request_id} got a conflicting duplicate admission bind; "
                "the original admission is preserved"
            )

        if num_external_tokens == 0:
            # HBM-complete admission returned (0, False): no Task-01 state, and
            # the parent must not fire its remote-prefill/metaserver flow.
            self._lookup_results.pop(request_id, None)
            return

        entry = self._lookup_results.pop(request_id, None)
        if entry is None:
            raise RuntimeError(f"DualPath request {request_id} has no admission lookup result to bind after allocation")
        local_tokens, cached_external_tokens, detached_spec = entry
        ready_tokens = max(request.num_tokens - 1, 0)
        transfer_tokens = self._hybrid_prefill_token_count(request.num_tokens)
        store_full = (
            detached_spec is not None
            and detached_spec.vllm_cached_tokens == local_tokens
            and detached_spec.kvpool_cached_tokens == ready_tokens
        )
        expected_external_tokens = (ready_tokens if store_full else transfer_tokens) - local_tokens
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

        coordinator = self._path_decision_coordinator
        request_key = DualPathRequestKey(coordinator.decode_engine_instance_id, request_id)
        decision_request = PathDecisionRequest(
            request_key=request_key,
            target_tokens=ready_tokens,
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
        message = build_remote_decode_message(
            self,
            request_id,
            remote_block_ids,
            snapshot,
            decision_metadata,
        )

        coordinator.register_pending(request_key)
        assert self._decision_timeout_seconds is not None
        state = DecodePathDecisionState(
            request_key=request_key,
            decision_request=decision_request,
            request=request,
            deadline=time.monotonic() + self._decision_timeout_seconds,
            status=DecodeDecisionStatus.PENDING,
        )
        self._decode_decision_states[request_id] = state

        if params.get("do_virtual") is not True:
            try:
                future = self.executor.submit(
                    self._access_metaserver,
                    url=params.get("metaserver", None),
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

    def _build_external_control_failure(
        self,
        request_id: str,
        snapshot: DecodeKVSnapshot,
        reason: DualPathControlFailureReason,
    ) -> DualPathControlFailureMetadata:
        block_size = self.block_size[0]
        assert snapshot.local_tokens % block_size == 0
        invalid_block_ids = snapshot.final_block_ids[0][snapshot.local_tokens // block_size :]
        assert invalid_block_ids
        return DualPathControlFailureMetadata(
            request_id=request_id,
            invalid_block_ids=invalid_block_ids,
            reason=reason,
        )

    def _activate_committed_decision(
        self,
        decision: PathDecision,
        metadata: DualPathConnectorMetadata,
    ) -> None:
        result = decision.result
        request_id = result.request_key.decode_request_id
        state = self._decode_decision_states.get(request_id)
        if state is None:
            return
        if state.status is DecodeDecisionStatus.PENDING:
            pass
        elif state.status in (
            DecodeDecisionStatus.COMMITTED,
            DecodeDecisionStatus.TIMED_OUT,
            DecodeDecisionStatus.ACTIVATION_FAILED,
        ):
            return
        else:
            assert_never(state.status)

        snapshot = self._decode_kv_snapshots[request_id]
        try:
            if decision.protocol_version != DUAL_PATH_PROTOCOL_VERSION:
                raise PathDecisionValidationError("Decision protocol version does not match the local protocol")
            if state.request_key != result.request_key or state.decision_request.request_key != result.request_key:
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
            binding = ForwardReceiveBinding(
                request_key=state.request_key,
                path=result.path,
                wire_request_id=get_external_request_id(request_id),
                decode_request_id=request_id,
                destination_block_ids=destination_block_ids,
                token_start=snapshot.local_tokens,
                token_end=snapshot.transfer_tokens,
            )
            reverse_plan: ReversePlan | None = None

            if result.path is Path.PE_READ:
                if decision.reverse_plan is not None:
                    raise PathDecisionValidationError("PE_READ Decision must not carry a Reverse plan")
            elif result.path is Path.DE_READ:
                reverse_plan = decision.reverse_plan
                if reverse_plan is None:
                    raise PathDecisionValidationError("DE_READ Decision requires a Reverse plan")
                if reverse_plan.request_key != state.request_key:
                    raise PathDecisionValidationError("Reverse plan key does not match the pending Decode state")
                if reverse_plan.wire_request_id != get_external_request_id(request_id):
                    raise PathDecisionValidationError("Reverse plan wire id does not match the Decode request")
                if reverse_plan.token_end != snapshot.store_tokens or reverse_plan.token_start >= snapshot.store_tokens:
                    raise PathDecisionValidationError("Reverse plan range does not end at the frozen Store boundary")
                if reverse_plan.source_block_ids != destination_block_ids:
                    raise PathDecisionValidationError("Reverse plan source does not match the advertised Decode table")
                if len(reverse_plan.remote_block_sizes) != len(self.block_size):
                    raise PathDecisionValidationError("Reverse plan block-size group count does not match Decode")
                if len(reverse_plan.destination_block_ids) != len(reverse_plan.remote_block_sizes):
                    raise PathDecisionValidationError("Reverse plan destination group count does not match block sizes")
                if not reverse_plan.remote_engine_id or not reverse_plan.remote_host or reverse_plan.remote_port <= 0:
                    raise PathDecisionValidationError("Reverse plan peer endpoint is invalid")
                if (
                    min(
                        reverse_plan.remote_tp_size,
                        reverse_plan.remote_pcp_size,
                        reverse_plan.remote_dcp_size,
                    )
                    <= 0
                ):
                    raise PathDecisionValidationError("Reverse plan topology is invalid")
                frozen_wrapper_blocks = tuple(tuple(group) for group in snapshot.allocated_blocks.get_block_ids())
                if frozen_wrapper_blocks != snapshot.final_block_ids:
                    raise PathDecisionValidationError("retained allocation wrapper no longer matches the snapshot")

                binding = ForwardReceiveBinding(
                    request_key=state.request_key,
                    path=Path.DE_READ,
                    wire_request_id=get_external_request_id(request_id),
                    decode_request_id=request_id,
                    destination_block_ids=destination_block_ids,
                    token_start=snapshot.store_tokens,
                    token_end=snapshot.transfer_tokens,
                )
                if snapshot.store_load_spec is not None:
                    assert self._kvpool_adapter is not None
                    self._kvpool_adapter.commit_after_alloc(
                        state.request,
                        snapshot.allocated_blocks,
                        snapshot.store_load_spec,
                    )
            else:
                assert_never(result.path)
        except Exception as error:  # noqa: BLE001
            logger.error("DualPath Decode activation failed for request %s: %s", request_id, error)
            state.status = DecodeDecisionStatus.ACTIVATION_FAILED
            self._path_decision_coordinator.unregister(state.request_key)
            metadata.control_failures.append(
                self._build_external_control_failure(
                    request_id,
                    snapshot,
                    DualPathControlFailureReason.ACTIVATION_FAILED,
                )
            )
            return

        if reverse_plan is not None:
            metadata.reverse_plans.append(reverse_plan)
        metadata.forward_receive_bindings.append(binding)
        state.status = DecodeDecisionStatus.COMMITTED
        if result.path is Path.PE_READ:
            store_coverage = "none"
            store_end = snapshot.local_tokens
            reverse_start = snapshot.local_tokens
            reverse_end = snapshot.local_tokens
        elif result.path is Path.DE_READ:
            store_coverage = "partial" if snapshot.store_load_spec is not None else "skipped"
            store_end = snapshot.store_tokens
            assert reverse_plan is not None
            reverse_start = reverse_plan.token_start
            reverse_end = reverse_plan.token_end
        else:
            assert_never(result.path)
        logger.info(
            "dual_path activation key=%s/%s protocol=%s selected_path=%s store=%s "
            "store_range=[%s,%s) reverse_range=[%s,%s) compute_range=[%s,%s) forward_range=[%s,%s)",
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
            binding.token_start,
            binding.token_end,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> MooncakeLayerwiseConnectorMetadata:
        parent_metadata = super().build_connector_meta(scheduler_output)
        if self.dual_path_cfg.role != "decode":
            self._sweep_pe_delivery()
            if not self._pe_pending_reverse_receive_bindings and not self._pe_control_failures:
                return parent_metadata
            metadata = DualPathConnectorMetadata()
            metadata.requests = parent_metadata.requests
            metadata.send_task = parent_metadata.send_task
            metadata.reverse_receive_bindings.extend(self._pe_pending_reverse_receive_bindings.values())
            metadata.control_failures.extend(self._pe_control_failures.values())
            self._pe_pending_reverse_receive_bindings.clear()
            self._pe_control_failures.clear()
            return metadata

        metadata = DualPathConnectorMetadata()
        metadata.requests = parent_metadata.requests
        metadata.send_task = parent_metadata.send_task
        coordinator = self._path_decision_coordinator

        for decision in coordinator.take_received_decisions():
            request_key = decision.result.request_key
            if request_key.decode_engine_instance_id != coordinator.decode_engine_instance_id:
                continue
            self._activate_committed_decision(decision, metadata)

        now = time.monotonic()
        for request_id, state in self._decode_decision_states.items():
            if state.status is DecodeDecisionStatus.PENDING:
                if state.deadline > now:
                    continue
            elif state.status in (
                DecodeDecisionStatus.COMMITTED,
                DecodeDecisionStatus.TIMED_OUT,
                DecodeDecisionStatus.ACTIVATION_FAILED,
            ):
                continue
            else:
                assert_never(state.status)
            state.status = DecodeDecisionStatus.TIMED_OUT
            coordinator.unregister(state.request_key)
            snapshot = self._decode_kv_snapshots[request_id]
            metadata.control_failures.append(
                self._build_external_control_failure(
                    request_id,
                    snapshot,
                    DualPathControlFailureReason.DECISION_TIMEOUT,
                )
            )

        assert self._kvpool_adapter is not None
        store_metadata = self._kvpool_adapter.build_connector_meta(scheduler_output)
        attach_store_metadata = bool(
            store_metadata.requests
            or store_metadata.unfinished_request_ids
            or store_metadata.preempted_req_ids
            or store_metadata.loading_req_ids
            or store_metadata.delayed_free_req_ids
        )
        if attach_store_metadata:
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
            self._pe_prefill_local_tokens.pop(request_id, None)
            self._pe_path_results.pop(request_id, None)
            self._pe_forward_plans.pop(request_id, None)
            self._pe_pending_reverse_receive_bindings.pop(request_id, None)
            self._pe_control_failures.pop(request_id, None)
            owned_send_req_info = self._pe_forward_send_infos.pop(request_id, None)
            send_req_info = self._reqs_need_send_layerwise.get(request_id)
            if owned_send_req_info is send_req_info and send_req_info is not None and send_req_info.request is request:
                self._reqs_need_send_layerwise.pop(request_id)
            released_key = self._pe_request_keys.pop(request_id, None)
            self._pe_invalid_request_ids.discard(request_id)
            self._sweep_pe_delivery(released_key)

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict | None]:
        self._release_scheduler_request_state(request)
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict | None]:
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


class DualPathConnectorWorker(MooncakeLayerwiseConnectorWorker):
    """Worker side of DualPathConnector.

    A Decode-role worker owns a non-layerwise ``KVPoolWorkerAdapter`` for the
    local Store load lifecycle and its existing ``LookupKeyServer``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        engine_id: str,
        dual_path_cfg: DualPathConfig,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, engine_id)
        self.dual_path_cfg = dual_path_cfg
        self._kvpool_worker_adapter: KVPoolWorkerAdapter | None = None
        self._registered_kv_caches: dict[str, list[torch.Tensor]] | None = None
        self._registered_layer_order: tuple[tuple[int, str], ...] = ()
        self._accepting_split_requests = True
        self._split_trackers: dict[str, _SplitTracker] = {}
        self._reverse_plans: dict[str, ReversePlan] = {}
        self._reverse_terminal_lock = threading.Lock()
        self._pending_local_reverse_terminals: dict[str, bool] = {}
        self._control_failed_recving: set[str] = set()
        self._forward_receive_bindings: dict[str, ForwardReceiveBinding] = {}
        self._pending_forward_done: set[str] = set()
        self._pending_forward_failed: set[str] = set()
        self._consumed_forward_terminals: dict[str, str] = {}
        self._reverse_receive_bindings: dict[str, ReverseReceiveBinding] = {}
        self._reverse_request_map: dict[str, str] = {}
        self._pending_reverse_done: set[str] = set()
        self._pending_reverse_failed: set[str] = set()
        self._consumed_reverse_terminals: dict[str, bool] = {}
        if dual_path_cfg.role == "decode":
            self._kvpool_worker_adapter = KVPoolWorkerAdapter(vllm_config, kv_cache_config)
        logger.info(
            "Initializing DualPath Worker %s (role=%s)",
            engine_id,
            dual_path_cfg.role,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        super().register_kv_caches(kv_caches)
        self._registered_kv_caches = {
            layer_name: list(kv_cache) if isinstance(kv_cache, (list, tuple)) else [kv_cache]
            for layer_name, kv_cache in kv_caches.items()
        }
        self._registered_layer_order = tuple((index, names[0]) for index, names in sorted(self.index_to_name.items()))
        if self.dual_path_cfg.role == "prefill":
            self._ensure_receive_layer_runtime()
        else:
            self._ensure_send_layer_runtime()
        if self._kvpool_worker_adapter is not None:
            self._kvpool_worker_adapter.register_kv_caches(kv_caches)

    def _install_forward_receive_binding(self, binding: ForwardReceiveBinding) -> None:
        if binding.path is Path.DE_READ and not getattr(self, "_accepting_split_requests", True):
            return
        consumed_decode_request_id = self._consumed_forward_terminals.get(binding.wire_request_id)
        if consumed_decode_request_id is not None:
            if consumed_decode_request_id == binding.decode_request_id:
                return
            raise RuntimeError(
                f"DualPath wire request {binding.wire_request_id} already belongs to a consumed Forward terminal; "
                "the original binding is preserved"
            )

        existing_binding = self._forward_receive_bindings.get(binding.decode_request_id)
        existing_decode_request_id = self.request_map.get(binding.wire_request_id)
        conflicts_with_retained_binding = any(
            retained != binding
            and (retained.request_key == binding.request_key or retained.wire_request_id == binding.wire_request_id)
            for retained in self._forward_receive_bindings.values()
        )
        if (
            (existing_binding is not None and existing_binding != binding)
            or (existing_decode_request_id is not None and existing_decode_request_id != binding.decode_request_id)
            or conflicts_with_retained_binding
        ):
            raise RuntimeError(
                f"DualPath Decode request {binding.decode_request_id} got a conflicting duplicate "
                "Forward receive binding; the original binding is preserved"
            )

        self.request_map[binding.wire_request_id] = binding.decode_request_id
        self._forward_receive_bindings[binding.decode_request_id] = binding

    def _release_finished_forward_terminals(self, finished_req_ids: set[str]) -> set[str]:
        finished_wire_ids = {get_external_request_id(request_id) for request_id in finished_req_ids}
        for wire_request_id in finished_wire_ids:
            decode_request_id = self._consumed_forward_terminals.get(wire_request_id)
            if decode_request_id in finished_req_ids:
                self._consumed_forward_terminals.pop(wire_request_id)
        self._pending_forward_done.difference_update(finished_wire_ids)
        self._pending_forward_failed.difference_update(finished_wire_ids)
        return finished_wire_ids

    def _install_reverse_receive_binding(self, binding: ReverseReceiveBinding) -> None:
        if not getattr(self, "_accepting_split_requests", True):
            return
        existing_binding = self._reverse_receive_bindings.get(binding.prefill_request_id)
        retained_wire_binding = next(
            (
                retained
                for retained in self._reverse_receive_bindings.values()
                if retained.wire_request_id == binding.wire_request_id
            ),
            None,
        )
        existing_prefill_request_id = self._reverse_request_map.get(binding.wire_request_id)
        existing_parent_request_id = self.request_map.get(binding.wire_request_id)
        conflicts_with_retained_binding = any(
            retained != binding
            and (retained.request_key == binding.request_key or retained.wire_request_id == binding.wire_request_id)
            for retained in self._reverse_receive_bindings.values()
        )
        if (
            (existing_binding is not None and existing_binding != binding)
            or (retained_wire_binding is not None and retained_wire_binding != binding)
            or (existing_prefill_request_id is not None and existing_prefill_request_id != binding.prefill_request_id)
            or existing_parent_request_id is not None
            or binding.wire_request_id in self._consumed_forward_terminals
            or conflicts_with_retained_binding
        ):
            raise RuntimeError(
                f"DualPath Prefill request {binding.prefill_request_id} got a conflicting duplicate "
                "Reverse receive binding; the original binding is preserved"
            )
        if binding.wire_request_id in self._consumed_reverse_terminals:
            if retained_wire_binding == binding:
                return
            raise RuntimeError(
                f"DualPath wire request {binding.wire_request_id} already belongs to a consumed Reverse terminal; "
                "the original binding is preserved"
            )

        self._reverse_request_map[binding.wire_request_id] = binding.prefill_request_id
        self._reverse_receive_bindings[binding.prefill_request_id] = binding
        if binding.wire_request_id in self._pending_forward_done:
            self._pending_forward_done.remove(binding.wire_request_id)
            self._pending_reverse_done.add(binding.wire_request_id)
        if binding.wire_request_id in self._pending_forward_failed:
            self._pending_forward_failed.remove(binding.wire_request_id)
            self._pending_reverse_failed.add(binding.wire_request_id)

    def _release_split_request_state(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_wire_ids = self._release_finished_forward_terminals(finished_req_ids)
        finished_reverse_wire_ids = self._release_finished_reverse_terminals(finished_req_ids)
        self._control_failed_recving.difference_update(finished_req_ids)
        split_trackers = getattr(self, "_split_trackers", {})
        reverse_plans = getattr(self, "_reverse_plans", {})
        for request_id in finished_req_ids:
            binding = self._forward_receive_bindings.get(request_id)
            if binding is not None:
                if self.request_map.get(binding.wire_request_id) == request_id:
                    self.request_map.pop(binding.wire_request_id)
                self._forward_receive_bindings.pop(request_id)
            reverse_plans.pop(request_id, None)
        reverse_terminal_lock = getattr(self, "_reverse_terminal_lock", None)
        if reverse_terminal_lock is not None:
            with reverse_terminal_lock:
                for request_id in finished_req_ids:
                    split_trackers.pop(request_id, None)
                    self._pending_local_reverse_terminals.pop(request_id, None)
        else:
            for request_id in finished_req_ids:
                split_trackers.pop(request_id, None)
                getattr(self, "_pending_local_reverse_terminals", {}).pop(request_id, None)
        return finished_wire_ids, finished_reverse_wire_ids

    def _release_finished_reverse_terminals(self, finished_req_ids: set[str]) -> set[str]:
        finished_wire_ids: set[str] = set()
        reverse_receive_bindings = getattr(self, "_reverse_receive_bindings", {})
        reverse_request_map = getattr(self, "_reverse_request_map", {})
        consumed_reverse_terminals = getattr(self, "_consumed_reverse_terminals", {})
        for prefill_request_id in finished_req_ids:
            binding = reverse_receive_bindings.pop(prefill_request_id, None)
            if binding is None:
                continue
            finished_wire_ids.add(binding.wire_request_id)
            reverse_request_map.pop(binding.wire_request_id, None)
            consumed_reverse_terminals.pop(binding.wire_request_id, None)
        getattr(self, "_pending_reverse_done", set()).difference_update(finished_wire_ids)
        getattr(self, "_pending_reverse_failed", set()).difference_update(finished_wire_ids)
        return finished_wire_ids

    def _consume_reverse_receive_binding(self, binding: ReverseReceiveBinding, terminal_flag: bool) -> None:
        self._consumed_reverse_terminals[binding.wire_request_id] = terminal_flag
        self._reverse_request_map.pop(binding.wire_request_id, None)

    def _consume_forward_receive_binding(self, binding: ForwardReceiveBinding) -> None:
        self._consumed_forward_terminals[binding.wire_request_id] = binding.decode_request_id
        self.request_map.pop(binding.wire_request_id, None)
        self._forward_receive_bindings.pop(binding.decode_request_id, None)

    def _install_split_tracker(
        self,
        binding: ForwardReceiveBinding,
        store_metadata: AscendConnectorMetadata | None,
    ) -> None:
        if not getattr(self, "_accepting_split_requests", True):
            return
        if binding.decode_request_id in self._split_trackers:
            return

        store_request = None
        if store_metadata is not None:
            store_request = next(
                (
                    request
                    for request in store_metadata.requests
                    if request.req_id == binding.decode_request_id and request.load_spec is not None
                ),
                None,
            )

        block_size = self.block_size[0]
        first_forward_block = binding.token_start // block_size
        last_forward_block = math.ceil(binding.token_end / block_size)
        forward_destination_slice = tuple(binding.destination_block_ids[0][first_forward_block:last_forward_block])
        if store_request is None:
            store_phase = _SplitPhase.SKIPPED
            store_destination_slice = ()
        else:
            load_spec = store_request.load_spec
            assert load_spec is not None
            first_store_block = load_spec.vllm_cached_tokens // block_size
            last_store_block = math.ceil(load_spec.kvpool_cached_tokens / block_size)
            store_phase = _SplitPhase.PENDING
            store_destination_slice = tuple(binding.destination_block_ids[0][first_store_block:last_store_block])

        self._split_trackers[binding.decode_request_id] = _SplitTracker(
            store_phase=store_phase,
            reverse_phase=_SplitPhase.SKIPPED,
            forward_phase=_SplitPhase.PENDING,
            store_destination_slice=store_destination_slice,
            forward_destination_slice=forward_destination_slice,
            plan=None,
            reverse_submitted=False,
            terminal_published=False,
        )

    def _install_reverse_plan(self, plan: ReversePlan) -> None:
        if not getattr(self, "_accepting_split_requests", True):
            return
        decode_request_id = plan.request_key.decode_request_id
        if get_external_request_id(decode_request_id) != plan.wire_request_id:
            raise RuntimeError(
                f"DualPath Decode request {decode_request_id} got a Reverse plan whose wire request id "
                "does not match the Decode-local request id"
            )

        binding = self._forward_receive_bindings.get(decode_request_id)
        tracker = self._split_trackers.get(decode_request_id)
        if binding is None or tracker is None:
            raise RuntimeError(
                f"DualPath Decode request {decode_request_id} got a Reverse plan before its split binding"
            )
        if binding.request_key != plan.request_key:
            raise RuntimeError(
                f"DualPath Decode request {decode_request_id} got a Reverse plan for a different request key"
            )
        if plan.token_end != binding.token_start:
            raise RuntimeError(
                f"DualPath Decode request {decode_request_id} got a Reverse plan with a conflicting split boundary"
            )

        existing_plan = self._reverse_plans.get(decode_request_id)
        conflicts_with_retained_plan = any(
            retained != plan
            and (retained.request_key == plan.request_key or retained.wire_request_id == plan.wire_request_id)
            for retained in self._reverse_plans.values()
        )
        if (existing_plan is not None and existing_plan != plan) or conflicts_with_retained_plan:
            raise RuntimeError(
                f"DualPath Decode request {decode_request_id} got a conflicting duplicate Reverse plan; "
                "the original plan is preserved"
            )
        if existing_plan is not None:
            return

        self._reverse_plans[decode_request_id] = plan
        tracker.plan = plan
        tracker.reverse_phase = _SplitPhase.PENDING

    def _build_reverse_send_metadata(
        self,
        plan: ReversePlan,
        decode_request_id: str,
    ) -> MooncakeLayerwiseConnectorMetadata:
        assert self.pd_head_ratio == 1 and not self.enable_kv_quant and not self.enable_c8_quant, (
            "DualPath Reverse supports the plain Layerwise send path only"
        )
        metadata = MooncakeLayerwiseConnectorMetadata()
        req_meta = ReqMeta(
            local_block_ids=[list(group) for group in plan.source_block_ids],
            token_ids=None,
            remote_block_ids=[list(group) for group in plan.destination_block_ids],
            remote_block_size=list(plan.remote_block_sizes),
            remote_engine_id=plan.remote_engine_id,
            remote_host=plan.remote_host,
            remote_port=plan.remote_port,
            remote_te_rpc_port=None,
            remote_layer_metadata=None,
            metaserver=None,
            remote_tp_size=plan.remote_tp_size,
            remote_pcp_size=plan.remote_pcp_size,
            remote_dcp_size=plan.remote_dcp_size,
            chunk_finish=False,
            prompt_len=plan.token_end,
            trans_count=[],
            remote_cache_tokens=0,
            local_computed_tokens=plan.token_end,
            local_transed_tokens=plan.token_start,
            do_virtual=False,
        )
        self._align_remote_block_ids(req_meta)
        transfer_mappings: dict[tuple[str, int], dict[str, Any]] = {}
        for group_idx in range(self.num_kv_cache_groups):
            group_mappings = self._get_kv_split_metadata(req_meta, 0, decode_request_id, group_idx)
            for (host, port), block_mapping in group_mappings.items():
                if (host, port) not in transfer_mappings:
                    transfer_mappings[(host, port)] = {
                        "local_block_ids": [[] for _ in range(self.num_kv_cache_groups)],
                        "remote_block_ids": [[] for _ in range(self.num_kv_cache_groups)],
                        "trans_count": [0 for _ in range(self.num_kv_cache_groups)],
                    }
                transfer_mappings[(host, port)]["local_block_ids"][group_idx].extend(block_mapping["local_block_ids"])
                transfer_mappings[(host, port)]["remote_block_ids"][group_idx].extend(block_mapping["remote_block_ids"])
                transfer_mappings[(host, port)]["trans_count"][group_idx] = block_mapping["trans_count"]

        assert len(transfer_mappings) <= 1, f"Not support add mutil transfer task for req_id:{decode_request_id}"
        for (host, port), block_mapping in transfer_mappings.items():
            update_req_meta = copy.deepcopy(req_meta)
            update_req_meta.remote_host = host
            update_req_meta.remote_port = port
            update_req_meta.local_block_ids = self._get_kernel_block_ids(block_mapping["local_block_ids"])
            update_req_meta.remote_block_ids = self._get_kernel_block_ids(block_mapping["remote_block_ids"])
            update_req_meta.trans_count = block_mapping["trans_count"]
            metadata.requests[decode_request_id] = update_req_meta
        return metadata

    def _submit_reverse(self, decode_request_id: str) -> None:
        if not getattr(self, "_accepting_split_requests", True):
            return
        tracker = self._split_trackers.get(decode_request_id)
        if (
            tracker is None
            or tracker.store_phase not in {_SplitPhase.DONE, _SplitPhase.SKIPPED}
            or tracker.plan is None
            or tracker.reverse_submitted
        ):
            return

        metadata = self._build_reverse_send_metadata(tracker.plan, decode_request_id)
        assert self._registered_kv_caches is not None
        ready_event = torch.npu.Event()
        ready_event.record()
        tracker.reverse_submitted = True
        for layer_index, layer_name in self._registered_layer_order:
            self._enqueue_kv_layer_send(
                layer_index=layer_index,
                layer_name=layer_name,
                kv_layer=self._registered_kv_caches[layer_name],
                ready_event=ready_event,
                metadata=metadata,
            )

    def _consume_store_completions(
        self,
        store_done_recving: set[str],
        store_invalid_block_ids: set[int],
    ) -> set[str]:
        published_store_terminals: set[str] = set()
        split_trackers = getattr(self, "_split_trackers", {})
        for request_id in store_done_recving:
            tracker = split_trackers.get(request_id)
            if tracker is None:
                published_store_terminals.add(request_id)
                continue
            if tracker.store_phase is not _SplitPhase.PENDING:
                continue
            if store_invalid_block_ids.intersection(tracker.store_destination_slice):
                tracker.store_phase = _SplitPhase.FAILED
                self._invalid_block_ids.update(tracker.store_destination_slice)
                logger.warning(
                    "dual_path data_terminal key=%s failure_source=STORE final_predicate=FAILED",
                    request_id,
                )
                if not tracker.terminal_published:
                    tracker.terminal_published = True
                    published_store_terminals.add(request_id)
            else:
                tracker.store_phase = _SplitPhase.DONE
                self._submit_reverse(request_id)
                if (
                    not tracker.terminal_published
                    and tracker.reverse_phase in {_SplitPhase.SKIPPED, _SplitPhase.DONE}
                    and tracker.forward_phase is _SplitPhase.DONE
                ):
                    tracker.terminal_published = True
                    published_store_terminals.add(request_id)
        return published_store_terminals

    def start_load_kv(self, metadata: MooncakeLayerwiseConnectorMetadata) -> None:
        store_metadata = getattr(metadata, "decode_store_metadata", None)
        for binding in getattr(metadata, "reverse_receive_bindings", ()):
            if self.dual_path_cfg.role == "prefill":
                self._install_reverse_receive_binding(binding)
        for binding in getattr(metadata, "forward_receive_bindings", ()):
            self._install_forward_receive_binding(binding)
            if binding.path is Path.DE_READ and self.dual_path_cfg.role == "decode":
                self._install_split_tracker(binding, store_metadata)
        for plan in getattr(metadata, "reverse_plans", ()):
            if self.dual_path_cfg.role == "decode":
                self._install_reverse_plan(plan)
        for failure in getattr(metadata, "control_failures", ()):
            self._control_failed_recving.add(failure.request_id)
            self._invalid_block_ids.update(failure.invalid_block_ids)
            logger.warning(
                "dual_path control_terminal key=%s failure_source=%s final_predicate=FAILED",
                failure.request_id,
                failure.reason.value,
            )
        split_store_accepted = getattr(self, "_accepting_split_requests", True) or not any(
            binding.path is Path.DE_READ for binding in getattr(metadata, "forward_receive_bindings", ())
        )
        if store_metadata is not None and split_store_accepted:
            assert self._kvpool_worker_adapter is not None
            self._kvpool_worker_adapter.start_load_kv(store_metadata)
        for binding in getattr(metadata, "forward_receive_bindings", ()):
            if binding.path is Path.DE_READ and self.dual_path_cfg.role == "decode":
                self._submit_reverse(binding.decode_request_id)
        super().start_load_kv(metadata)

    def send_done_send_signal(self, req_id, req_meta, group_idx, trans_flag: bool = True):
        if self.dual_path_cfg.role == "decode":
            with self._reverse_terminal_lock:
                tracker = self._split_trackers.get(req_id)
                if tracker is not None and tracker.reverse_submitted:
                    self._pending_local_reverse_terminals[req_id] = (
                        self._pending_local_reverse_terminals.get(req_id, True) and trans_flag
                    )
        super().send_done_send_signal(req_id, req_meta, group_idx, trans_flag)

    def get_finished(
        self,
        finished_req_ids: set[str],
        metadata: MooncakeLayerwiseConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        finished_wire_ids, finished_reverse_wire_ids = self._release_split_request_state(finished_req_ids)
        split_trackers = getattr(self, "_split_trackers", {})

        done_sending: set[str] = set()
        done_recving: set[str] = set()
        store_metadata = getattr(metadata, "decode_store_metadata", None)
        if store_metadata is not None:
            assert self._kvpool_worker_adapter is not None
            store_done_sending, store_done_recving = self._kvpool_worker_adapter.get_finished(
                finished_req_ids,
                store_metadata,
            )
            store_invalid_block_ids = self._kvpool_worker_adapter.get_block_ids_with_load_errors()
            self._invalid_block_ids.update(store_invalid_block_ids)
            done_sending.update(store_done_sending)
            done_recving.update(self._consume_store_completions(store_done_recving, store_invalid_block_ids))

        reverse_terminal_lock = getattr(self, "_reverse_terminal_lock", None)
        if reverse_terminal_lock is None:
            local_reverse_terminals = {}
        else:
            with reverse_terminal_lock:
                local_reverse_terminals = dict(self._pending_local_reverse_terminals)
                self._pending_local_reverse_terminals.clear()
        for request_id, terminal_flag in local_reverse_terminals.items():
            tracker = split_trackers.get(request_id)
            if tracker is None or tracker.reverse_phase is not _SplitPhase.PENDING:
                continue
            if terminal_flag:
                tracker.reverse_phase = _SplitPhase.DONE
                if (
                    not tracker.terminal_published
                    and tracker.store_phase in {_SplitPhase.SKIPPED, _SplitPhase.DONE}
                    and tracker.forward_phase is _SplitPhase.DONE
                ):
                    tracker.terminal_published = True
                    done_recving.add(request_id)
            else:
                tracker.reverse_phase = _SplitPhase.FAILED
                self._invalid_block_ids.update(tracker.forward_destination_slice)
                logger.warning(
                    "dual_path data_terminal key=%s failure_source=REVERSE final_predicate=FAILED",
                    request_id,
                )
                if not tracker.terminal_published:
                    tracker.terminal_published = True
                    done_recving.add(request_id)

        if self.kv_recv_layer_thread is not None:
            raw_done = self.kv_recv_layer_thread.get_and_clear_done_requests()
            raw_failed = self.kv_recv_layer_thread.get_and_clear_failed_requests()
        else:
            raw_done = set()
            raw_failed = set()

        reverse_request_map = getattr(self, "_reverse_request_map", {})
        reverse_receive_bindings = getattr(self, "_reverse_receive_bindings", {})
        pending_reverse_done = getattr(self, "_pending_reverse_done", set())
        pending_reverse_failed = getattr(self, "_pending_reverse_failed", set())
        ignored_wire_ids = set(self._consumed_forward_terminals).union(getattr(self, "_consumed_reverse_terminals", {}))
        ignored_wire_ids.update(
            wire_request_id for wire_request_id in finished_wire_ids if wire_request_id not in self.request_map
        )
        ignored_wire_ids.update(
            wire_request_id
            for wire_request_id in finished_reverse_wire_ids
            if wire_request_id not in reverse_request_map
        )
        raw_done.difference_update(ignored_wire_ids)
        raw_failed.difference_update(ignored_wire_ids)

        reverse_owned_wire_ids = set(reverse_request_map)
        reverse_done_wire_ids = raw_done.intersection(reverse_owned_wire_ids).union(pending_reverse_done)
        reverse_failed_wire_ids = raw_failed.intersection(reverse_owned_wire_ids).union(pending_reverse_failed)
        raw_done.difference_update(reverse_owned_wire_ids)
        raw_failed.difference_update(reverse_owned_wire_ids)
        reverse_done_wire_ids.difference_update(reverse_failed_wire_ids)
        reverse_finished: set[str] = set()
        for wire_request_id in reverse_failed_wire_ids:
            prefill_request_id = reverse_request_map[wire_request_id]
            binding = reverse_receive_bindings[prefill_request_id]
            first_reverse_block = binding.token_start // self.block_size[0]
            last_reverse_block = math.ceil(binding.token_end / self.block_size[0])
            self._invalid_block_ids.update(binding.destination_block_ids[0][first_reverse_block:last_reverse_block])
            reverse_finished.add(prefill_request_id)
            self._consume_reverse_receive_binding(binding, False)
            logger.warning(
                "dual_path data_terminal key=%s failure_source=REVERSE final_predicate=FAILED",
                prefill_request_id,
            )
        for wire_request_id in reverse_done_wire_ids:
            prefill_request_id = reverse_request_map[wire_request_id]
            binding = reverse_receive_bindings[prefill_request_id]
            reverse_finished.add(prefill_request_id)
            self._consume_reverse_receive_binding(binding, True)
            logger.info(
                "dual_path reverse_terminal key=%s terminal=DONE final_predicate=SUCCESS",
                prefill_request_id,
            )
        pending_reverse_done.clear()
        pending_reverse_failed.clear()

        done_wire_ids = raw_done.union(self._pending_forward_done)
        failed_wire_ids = raw_failed.union(self._pending_forward_failed)
        failed_wins_wire_ids: set[str] = set()
        for wire_request_id in failed_wire_ids:
            decode_request_id = self.request_map.get(wire_request_id)
            if decode_request_id is None:
                failed_wins_wire_ids.add(wire_request_id)
                continue
            binding = self._forward_receive_bindings.get(decode_request_id)
            if binding is not None and binding.wire_request_id == wire_request_id:
                failed_wins_wire_ids.add(wire_request_id)
        done_wire_ids.difference_update(failed_wins_wire_ids)
        pending_done: set[str] = set()
        pending_failed: set[str] = set()
        ordinary_done: set[str] = set()
        ordinary_failed: set[str] = set()
        forward_finished: set[str] = set()

        for wire_request_id in failed_wire_ids:
            decode_request_id = self.request_map.get(wire_request_id)
            if decode_request_id is None:
                pending_failed.add(wire_request_id)
                continue
            binding = self._forward_receive_bindings.get(decode_request_id)
            if binding is None or binding.wire_request_id != wire_request_id:
                ordinary_failed.add(decode_request_id)
                continue
            if binding.path is Path.PE_READ:
                first_forward_block = binding.token_start // self.block_size[0]
                last_forward_block = math.ceil(binding.token_end / self.block_size[0])
                self._invalid_block_ids.update(binding.destination_block_ids[0][first_forward_block:last_forward_block])
                forward_finished.add(decode_request_id)
            elif binding.path is Path.DE_READ:
                tracker = split_trackers[decode_request_id]
                if tracker.forward_phase is _SplitPhase.PENDING:
                    tracker.forward_phase = _SplitPhase.FAILED
                    self._invalid_block_ids.update(tracker.forward_destination_slice)
                    logger.warning(
                        "dual_path data_terminal key=%s failure_source=FORWARD final_predicate=FAILED",
                        decode_request_id,
                    )
                    if not tracker.terminal_published:
                        tracker.terminal_published = True
                        forward_finished.add(decode_request_id)
            else:
                assert_never(binding.path)
            self._consume_forward_receive_binding(binding)

        for wire_request_id in done_wire_ids:
            decode_request_id = self.request_map.get(wire_request_id)
            if decode_request_id is None:
                pending_done.add(wire_request_id)
                continue
            binding = self._forward_receive_bindings.get(decode_request_id)
            if binding is None or binding.wire_request_id != wire_request_id:
                ordinary_done.add(decode_request_id)
                continue
            if binding.path is Path.PE_READ:
                forward_finished.add(decode_request_id)
            elif binding.path is Path.DE_READ:
                tracker = split_trackers[decode_request_id]
                if tracker.forward_phase is _SplitPhase.PENDING:
                    tracker.forward_phase = _SplitPhase.DONE
                    if (
                        not tracker.terminal_published
                        and tracker.store_phase in {_SplitPhase.SKIPPED, _SplitPhase.DONE}
                        and tracker.reverse_phase in {_SplitPhase.SKIPPED, _SplitPhase.DONE}
                    ):
                        tracker.terminal_published = True
                        forward_finished.add(decode_request_id)
            else:
                assert_never(binding.path)
            self._consume_forward_receive_binding(binding)

        for decode_request_id in ordinary_failed:
            if metadata := self._recving_metadata.get(decode_request_id):
                self._invalid_block_ids.update(block_id for group in metadata.local_block_ids for block_id in group)
        for decode_request_id in ordinary_done.union(ordinary_failed):
            self.request_map.pop(get_external_request_id(decode_request_id), None)
            self._recving_metadata.pop(decode_request_id, None)

        self._pending_forward_done = pending_done
        self._pending_forward_failed = pending_failed
        done_recving.update(ordinary_done.union(forward_finished, reverse_finished, self.virtual_request))
        self.virtual_request = set()
        for request_id in done_recving:
            tracker = split_trackers.get(request_id)
            if tracker is None or not tracker.terminal_published:
                continue
            final_predicate = (
                "FAILED"
                if _SplitPhase.FAILED in {tracker.store_phase, tracker.reverse_phase, tracker.forward_phase}
                else "SUCCESS"
            )
            logger.info(
                "dual_path final key=%s store=%s reverse=%s forward=%s final_predicate=%s",
                request_id,
                tracker.store_phase.value,
                tracker.reverse_phase.value,
                tracker.forward_phase.value,
                final_predicate,
            )
        if done_recving:
            logger.info(
                "Number of completed KV cache recv requests: %s, receive requests: %s",
                len(done_recving),
                done_recving,
            )
        done_recving.update(self._control_failed_recving)
        self._control_failed_recving.clear()
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_block_ids = super().get_block_ids_with_load_errors()
        if self._kvpool_worker_adapter is not None:
            invalid_block_ids.update(self._kvpool_worker_adapter.get_block_ids_with_load_errors())
        return invalid_block_ids

    def shutdown(self) -> None:
        """Release local state without promising cancellation of remote DMA."""
        if not getattr(self, "_accepting_split_requests", True):
            return
        self._accepting_split_requests = False
        for binding in self._forward_receive_bindings.values():
            if self.request_map.get(binding.wire_request_id) == binding.decode_request_id:
                self.request_map.pop(binding.wire_request_id)
        self._forward_receive_bindings.clear()
        self._pending_forward_done.clear()
        self._pending_forward_failed.clear()
        self._consumed_forward_terminals.clear()
        getattr(self, "_reverse_plans", {}).clear()
        reverse_terminal_lock = getattr(self, "_reverse_terminal_lock", None)
        if reverse_terminal_lock is not None:
            with reverse_terminal_lock:
                getattr(self, "_split_trackers", {}).clear()
                self._pending_local_reverse_terminals.clear()
        else:
            getattr(self, "_split_trackers", {}).clear()
        getattr(self, "_reverse_receive_bindings", {}).clear()
        getattr(self, "_reverse_request_map", {}).clear()
        getattr(self, "_pending_reverse_done", set()).clear()
        getattr(self, "_pending_reverse_failed", set()).clear()
        getattr(self, "_consumed_reverse_terminals", {}).clear()
        self._control_failed_recving.clear()
        if self._kvpool_worker_adapter is not None:
            self._kvpool_worker_adapter.close()
        parent_shutdown = getattr(super(), "shutdown", None)
        if parent_shutdown is not None:
            parent_shutdown()


class DualPathConnector(MooncakeLayerwiseConnector, SupportsHMA):
    """Layerwise connector with DualPath admission and transfer routing.

    The connector constructs DualPath Scheduler and Worker subclasses while
    preserving the parent's accounting, metadata, transfer, completion,
    invalid-block, cleanup, and failure behavior for ordinary Layerwise
    workloads.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        # NOTE: do NOT call MooncakeLayerwiseConnector.__init__ — it hard-builds
        # the parent scheduler/worker. Call KVConnectorBase_V1.__init__ exactly
        # once and replicate only the facade state listed in the PR-00
        # construction contract.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeLayerwiseConnectorMetadata()
        dual_path_cfg = DualPathConfig.from_extra_config(
            vllm_config.kv_transfer_config.kv_connector_extra_config,
            vllm_config.kv_transfer_config,
        )

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeLayerwiseConnectorScheduler | None = DualPathConnectorScheduler(
                vllm_config, kv_cache_config, str(self.engine_id), dual_path_cfg
            )
            self.connector_worker: MooncakeLayerwiseConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = DualPathConnectorWorker(
                vllm_config, kv_cache_config, str(self.engine_id), dual_path_cfg
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role!r}")

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert isinstance(self.connector_worker, DualPathConnectorWorker)
        return self.connector_worker.get_finished(finished_req_ids, self._connector_metadata)

    def shutdown(self):
        """Release DualPath-owned state and adapters, then defer to the base."""
        if isinstance(self.connector_scheduler, DualPathConnectorScheduler):
            self.connector_scheduler.shutdown()
        if isinstance(self.connector_worker, DualPathConnectorWorker):
            self.connector_worker.shutdown()
        super().shutdown()

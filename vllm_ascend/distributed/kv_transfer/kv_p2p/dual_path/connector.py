# SPDX-License-Identifier: Apache-2.0
"""DualPathConnector — behavior-preserving alias of MooncakeLayerwiseConnector.

PR-00 foundation scope: this connector inherits ``MooncakeLayerwiseConnector``
unchanged in behavior, so an ordinary Layerwise workload runs exactly as it
does today. It exists to create safe Scheduler and Worker subclass seams for
later PRs; it adds no DualPath decision or data path.

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

import math
import time
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, assert_never

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
    DecisionTimeoutMetadata,
    DualPathConnectorMetadata,
    ForwardPlan,
    ForwardReceiveBinding,
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


@dataclass(frozen=True)
class DecodeKVSnapshot:
    """Complete Task-01 admission record, created only after real allocation.

    ``local_tokens`` retains the original HBM prefix while ``store_tokens`` is
    derived from the detached ``LoadSpec``. ``final_block_ids`` is mandatory
    because the record cannot exist before vLLM allocates the final Decode blocks.
    """

    target_tokens: int
    transfer_tokens: int
    local_tokens: int
    external_tokens: int
    store_load_spec: LoadSpec | None
    final_block_ids: tuple[tuple[int, ...], ...]

    @property
    def store_tokens(self) -> int:
        if self.store_load_spec is None:
            return self.local_tokens
        return self.store_load_spec.kvpool_cached_tokens


class DecodeDecisionStatus(str, Enum):
    PENDING = "PENDING"
    COMMITTED = "COMMITTED"
    TIMED_OUT = "TIMED_OUT"


@dataclass
class DecodePathDecisionState:
    request_key: DualPathRequestKey
    decision_request: PathDecisionRequest
    deadline: float
    status: DecodeDecisionStatus
    result: PathDecisionResult | None = None
    proxy_future: Future[None] | None = None
    timeout_reported: bool = False


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

    Task-01: for a Decode-role request carrying ``do_remote_prefill=True``,
    owns the admission path from initial HBM lookup through final slot
    allocation: ``get_num_new_matched_tokens`` returns the route-dependent
    external delta ``E_DE = T - L_DE`` for remote-required requests (never the
    Store hit), and ``update_state_after_alloc`` binds the frozen final blocks
    into a ``DecodeKVSnapshot``. Task-04 adds the Prefill decision hook and
    suppresses inherited Forward queueing for its ``dual_path`` envelope.
    Requests outside those contracted shapes delegate to the parent unchanged.
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
        self._accepting_task01 = True
        self._accepting_pe_decisions = True
        # PE fields exist on both roles; Decode keeps a None decider and empty
        # maps rather than constructing policy state it never owns.
        self._path_decider: PathDecisionDecider | None = None
        self._pe_request_keys: dict[str, DualPathRequestKey] = {}
        self._pe_path_results: dict[str, PathDecisionResult] = {}
        self._pe_forward_plans: dict[str, ForwardPlan] = {}
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

    def _is_task01_decode_request(self, request: Request) -> bool:
        """Task-01 admission applies only to Decode-role requests that arrived
        with ``do_remote_prefill is True``; everything else keeps parent behavior."""
        if not self._accepting_task01 or self.dual_path_cfg.role != "decode":
            return False
        params = request.kv_transfer_params
        return params is not None and params.get("do_remote_prefill") is True

    def _sweep_pe_delivery(self, released_key: DualPathRequestKey | None = None) -> None:
        if self._path_decider is None:
            return

        active_keys = frozenset(self._pe_request_keys.values())
        sweep_keys = set(self._pe_delivery_futures)
        if released_key is not None:
            sweep_keys.add(released_key)

        for request_key in sweep_keys:
            if request_key in active_keys:
                continue
            delivery_future = self._pe_delivery_futures.get(request_key)
            if delivery_future is not None and not delivery_future.done():
                continue
            self._path_decider.discard(request_key)
            self._pe_delivery_futures.pop(request_key, None)

    def _handle_prefill_decision(
        self,
        request: Request,
        parent_result: tuple[int, bool],
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if (
            not self._accepting_pe_decisions
            or self.dual_path_cfg.role != "prefill"
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
        request_key = decision_request.request_key
        self._pe_request_keys[request_id] = request_key
        assert self._path_decider is not None
        try:
            result = self._path_decider.decide(decision_request)
        except Exception as error:  # noqa: BLE001
            logger.error(
                "DualPath Prefill decision failed locally for request %s: %s",
                request_id,
                error,
            )
            return parent_result

        retained_result = self._pe_path_results.get(request_id)
        if retained_result is None:
            self._pe_path_results[request_id] = result
        elif retained_result != result:
            raise RuntimeError(
                f"DualPath Prefill request {request_id} got a conflicting retained path result; "
                "the original result is preserved"
            )

        if request_key not in self._pe_delivery_futures:
            decision = PathDecision(
                protocol_version=DUAL_PATH_PROTOCOL_VERSION,
                result=result,
            )
            delivery_future = self._path_decision_coordinator.submit(
                metadata.decode_control_endpoint,
                decision,
            )
            self._pe_delivery_futures[request_key] = delivery_future

            def log_delivery_failure(completed_future: Future[None]) -> None:
                if completed_future.cancelled():
                    return
                error = completed_future.exception()
                if error is not None:
                    logger.error(
                        "DualPath Prefill decision delivery failed for request %s: %s",
                        request_id,
                        error,
                    )

            delivery_future.add_done_callback(log_delivery_failure)
        return parent_result

    def _try_install_forward_plan(self, request: Request, blocks: KVCacheBlocks) -> None:
        request_id = request.request_id
        result = self._pe_path_results.get(request_id)
        if result is None:
            return
        match result.path:
            case Path.DE_READ:
                return
            case Path.PE_READ:
                pass
            case unreachable:
                assert_never(unreachable)

        params = request.kv_transfer_params
        assert params is not None
        try:
            metadata = DualPathDecisionMetadata.from_dict(params["dual_path"])
            if metadata.protocol_version != DUAL_PATH_PROTOCOL_VERSION:
                raise PathDecisionValidationError(
                    f"protocol version must be {DUAL_PATH_PROTOCOL_VERSION}, got {metadata.protocol_version}"
                )
            decision_request = metadata.decision_request
            if result.request_key != decision_request.request_key:
                raise PathDecisionValidationError("retained result key does not match the Decision Request key")

            token_start = decision_request.decode_local_tokens
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

            required_fields = ("remote_block_size", "remote_engine_id", "remote_host", "remote_port")
            missing_fields = [field for field in required_fields if field not in params]
            if missing_fields:
                raise PathDecisionValidationError(f"missing inherited remote fields: {missing_fields}")
            if params.get("remote_cached_tokens") != token_start:
                raise PathDecisionValidationError(
                    "remote_cached_tokens does not match the Decision Request local prefix"
                )

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
            for group, block_size in zip(destination_block_ids, remote_block_sizes):
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
                len(group) * block_size < token_end for group, block_size in zip(source_block_ids, local_block_sizes)
            ):
                return

            plan = ForwardPlan(
                request_key=decision_request.request_key,
                token_start=token_start,
                token_end=token_end,
                source_block_ids=source_block_ids,
                destination_block_ids=destination_block_ids,
            )
        except (KeyError, PathDecisionValidationError, TypeError) as error:
            logger.error(
                "DualPath Prefill Forward plan is invalid for request %s: %s",
                request_id,
                error,
            )
            self._pe_invalid_request_ids.add(request_id)
            self._pe_path_results.pop(request_id, None)
            return

        existing_plan = self._pe_forward_plans.get(request_id)
        if existing_plan is not None:
            if existing_plan == plan:
                return
            raise RuntimeError(
                f"DualPath Prefill request {request_id} got a conflicting duplicate Forward plan; "
                "the original plan is preserved"
            )

        self._pe_forward_plans[request_id] = plan
        self._reqs_need_send_layerwise[request_id] = SendReqInfo(
            local_block_ids=[list(group) for group in plan.source_block_ids],
            local_transferred_tokens=plan.token_start,
            local_computed_tokens=0,
            request=request,
        )

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        if not self._is_task01_decode_request(request):
            parent_result = super().get_num_new_matched_tokens(request, num_computed_tokens)
            if self.dual_path_cfg.role == "prefill" and request.kv_transfer_params is not None:
                return self._handle_prefill_decision(request, parent_result)
            return parent_result

        request_id = request.request_id
        if request_id in self._decode_kv_snapshots:
            raise RuntimeError(
                f"DualPath request {request_id} is already admitted; a new initial lookup is a lifecycle error"
            )

        transfer_tokens = self._hybrid_prefill_token_count(request.num_tokens)
        ready_tokens = max(request.num_tokens - 1, 0)
        local_tokens = num_computed_tokens
        if local_tokens >= ready_tokens:
            # HBM-complete: no KVPool lookup, no Task-01 state.
            self._lookup_results.pop(request_id, None)
            return 0, False
        if not 0 <= local_tokens < ready_tokens <= transfer_tokens:
            raise RuntimeError(
                f"DualPath request {request_id} initial admission requires "
                f"0 <= local_tokens ({local_tokens}) < ready_tokens ({ready_tokens}) "
                f"<= transfer_tokens ({transfer_tokens})"
            )

        external_tokens = transfer_tokens - local_tokens
        cached = self._lookup_results.get(request_id)
        if cached is not None:
            if cached[0] == local_tokens and cached[1] == external_tokens:
                # Identical duplicate lookup (e.g. allocation-failure retry):
                # reuse the detached result instead of re-probing the KV pool.
                return external_tokens, True
            # A changed E_DE invalidates the unbound result; discard it before
            # the fresh lookup so a failing re-probe cannot leave it behind.
            del self._lookup_results[request_id]

        assert self._kvpool_adapter is not None
        detached_spec = self._kvpool_adapter.lookup(request, local_tokens)
        self._lookup_results[request_id] = (local_tokens, external_tokens, detached_spec)
        return external_tokens, True

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        params = request.kv_transfer_params
        if self.dual_path_cfg.role == "prefill" and params is not None and "dual_path" in params:
            self._try_install_forward_plan(request, blocks)
            return

        request_id = request.request_id
        target_tokens = max(request.num_tokens - 1, 0)
        transfer_tokens = self._hybrid_prefill_token_count(request.num_tokens)

        existing = self._decode_kv_snapshots.get(request_id)
        if existing is not None:
            allocated_block_ids = blocks.get_block_ids()
            frozen_block_ids = tuple(tuple(group) for group in allocated_block_ids)
            if (
                existing.target_tokens == target_tokens
                and existing.transfer_tokens == transfer_tokens
                and existing.local_tokens == transfer_tokens - num_external_tokens
                and existing.external_tokens == num_external_tokens
                and existing.final_block_ids == frozen_block_ids
            ):
                return
            raise RuntimeError(
                f"DualPath request {request_id} got a conflicting duplicate admission bind; "
                "the original admission is preserved"
            )

        if not self._is_task01_decode_request(request):
            return super().update_state_after_alloc(request, blocks, num_external_tokens)

        if num_external_tokens == 0:
            # HBM-complete admission returned (0, False): no Task-01 state, and
            # the parent must not fire its remote-prefill/metaserver flow.
            self._lookup_results.pop(request_id, None)
            return

        entry = self._lookup_results.pop(request_id, None)
        if entry is None:
            raise RuntimeError(f"DualPath request {request_id} has no Task-01 lookup result to bind after allocation")
        local_tokens, cached_external_tokens, detached_spec = entry
        expected_external_tokens = transfer_tokens - local_tokens
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
            if not local_tokens < detached_spec.kvpool_cached_tokens <= target_tokens:
                raise RuntimeError(
                    f"DualPath request {request_id} detached LoadSpec store tokens "
                    f"{detached_spec.kvpool_cached_tokens} outside ({local_tokens}, {target_tokens}]"
                )

        allocated_block_ids = blocks.get_block_ids()
        frozen_block_ids = tuple(tuple(group) for group in allocated_block_ids)
        snapshot = DecodeKVSnapshot(
            target_tokens=target_tokens,
            transfer_tokens=transfer_tokens,
            local_tokens=local_tokens,
            external_tokens=num_external_tokens,
            store_load_spec=detached_spec,
            final_block_ids=frozen_block_ids,
        )
        self._decode_kv_snapshots[request_id] = snapshot

        coordinator = self._path_decision_coordinator
        request_key = DualPathRequestKey(coordinator.decode_engine_instance_id, request_id)
        decision_request = PathDecisionRequest(
            request_key=request_key,
            target_tokens=snapshot.target_tokens,
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
            deadline=time.monotonic() + self._decision_timeout_seconds,
            status=DecodeDecisionStatus.PENDING,
        )
        self._decode_decision_states[request_id] = state

        params = request.kv_transfer_params
        assert params is not None
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
                state.proxy_future = future

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

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> MooncakeLayerwiseConnectorMetadata:
        parent_metadata = super().build_connector_meta(scheduler_output)
        if self.dual_path_cfg.role != "decode":
            self._sweep_pe_delivery()
            return parent_metadata

        metadata = DualPathConnectorMetadata()
        metadata.requests = parent_metadata.requests
        metadata.send_task = parent_metadata.send_task
        coordinator = self._path_decision_coordinator

        for result in coordinator.take_received_results():
            request_key = result.request_key
            if request_key.decode_engine_instance_id != coordinator.decode_engine_instance_id:
                continue
            state = self._decode_decision_states.get(request_key.decode_request_id)
            if state is None or state.request_key != request_key:
                continue
            match state.status:
                case DecodeDecisionStatus.PENDING:
                    match result.path:
                        case Path.DE_READ:
                            pass
                        case Path.PE_READ:
                            snapshot = self._decode_kv_snapshots[request_key.decode_request_id]
                            destination_block_ids = tuple(
                                tuple(group)
                                for group in self._trim_hybrid_remote_block_ids(
                                    snapshot.final_block_ids,
                                    snapshot.target_tokens + 1,
                                )
                            )
                            metadata.forward_receive_bindings.append(
                                ForwardReceiveBinding(
                                    request_key=state.request_key,
                                    wire_request_id=get_external_request_id(request_key.decode_request_id),
                                    decode_request_id=request_key.decode_request_id,
                                    destination_block_ids=destination_block_ids,
                                    token_start=snapshot.local_tokens,
                                    token_end=snapshot.transfer_tokens,
                                )
                            )
                        case unreachable:
                            assert_never(unreachable)
                    state.result = result
                    state.status = DecodeDecisionStatus.COMMITTED
                case DecodeDecisionStatus.COMMITTED | DecodeDecisionStatus.TIMED_OUT:
                    continue
                case unreachable:
                    assert_never(unreachable)

        now = time.monotonic()
        for request_id, state in self._decode_decision_states.items():
            match state.status:
                case DecodeDecisionStatus.PENDING:
                    if state.deadline > now:
                        continue
                case DecodeDecisionStatus.COMMITTED | DecodeDecisionStatus.TIMED_OUT:
                    continue
                case unreachable:
                    assert_never(unreachable)
            state.status = DecodeDecisionStatus.TIMED_OUT
            coordinator.unregister(state.request_key)
            snapshot = self._decode_kv_snapshots[request_id]
            block_size = self.block_size[0]
            assert snapshot.local_tokens % block_size == 0
            first_external_block = snapshot.local_tokens // block_size
            external_block_ids = snapshot.final_block_ids[0][first_external_block:]
            assert external_block_ids
            metadata.decision_timeouts.append(
                DecisionTimeoutMetadata(
                    request_id=request_id,
                    external_block_ids=tuple(external_block_ids),
                )
            )
            state.timeout_reported = True

        return metadata

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict | None]:
        request_id = request.request_id
        self._lookup_results.pop(request_id, None)
        self._decode_kv_snapshots.pop(request_id, None)
        state = self._decode_decision_states.pop(request_id, None)
        if state is not None:
            state.proxy_future = None
            self._path_decision_coordinator.unregister(state.request_key)
        if self.dual_path_cfg.role == "prefill":
            released_key = self._pe_request_keys.pop(request_id, None)
            self._pe_invalid_request_ids.discard(request_id)
            self._sweep_pe_delivery(released_key)
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict | None]:
        request_id = request.request_id
        self._lookup_results.pop(request_id, None)
        self._decode_kv_snapshots.pop(request_id, None)
        state = self._decode_decision_states.pop(request_id, None)
        if state is not None:
            state.proxy_future = None
            self._path_decision_coordinator.unregister(state.request_key)
        if self.dual_path_cfg.role == "prefill":
            released_key = self._pe_request_keys.pop(request_id, None)
            self._pe_invalid_request_ids.discard(request_id)
            self._sweep_pe_delivery(released_key)
        return super().request_finished_all_groups(request, block_ids)

    def shutdown(self) -> None:
        """Stop Task-04 work and release all owned records and clients."""
        self._accepting_task01 = False
        self._accepting_pe_decisions = False
        self._path_decision_coordinator.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.metaserver_client.close()
        for state in self._decode_decision_states.values():
            state.proxy_future = None
        if self._path_decider is not None:
            retained_keys = set(self._pe_request_keys.values())
            retained_keys.update(self._pe_delivery_futures)
            for request_key in retained_keys:
                self._path_decider.discard(request_key)
        self._lookup_results.clear()
        self._decode_kv_snapshots.clear()
        self._decode_decision_states.clear()
        self._pe_request_keys.clear()
        self._pe_delivery_futures.clear()
        self._pe_invalid_request_ids.clear()
        if self._kvpool_adapter is not None:
            self._kvpool_adapter.close()


class DualPathConnectorWorker(MooncakeLayerwiseConnectorWorker):
    """Worker side of DualPathConnector.

    Task-01: a Decode-role worker additionally owns a lookup-only
    ``KVPoolWorkerAdapter`` (non-layerwise ``KVPoolWorker`` + the existing
    ``LookupKeyServer`` on the owning rank). No transfer threads, KV cache
    registration, or Store load metadata are started by this adapter.
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
        self._control_failed_recving: set[str] = set()
        self._forward_receive_bindings: dict[str, ForwardReceiveBinding] = {}
        self._pending_forward_done: set[str] = set()
        self._pending_forward_failed: set[str] = set()
        self._consumed_forward_terminals: dict[str, str] = {}
        if dual_path_cfg.role == "decode":
            self._kvpool_worker_adapter = KVPoolWorkerAdapter(vllm_config, kv_cache_config)
        logger.info(
            "Initializing DualPath Worker %s (role=%s)",
            engine_id,
            dual_path_cfg.role,
        )

    def _install_forward_receive_binding(self, binding: ForwardReceiveBinding) -> None:
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

    def _consume_forward_receive_binding(self, binding: ForwardReceiveBinding) -> None:
        self._consumed_forward_terminals[binding.wire_request_id] = binding.decode_request_id
        self.request_map.pop(binding.wire_request_id, None)
        self._forward_receive_bindings.pop(binding.decode_request_id, None)

    def start_load_kv(self, metadata: MooncakeLayerwiseConnectorMetadata) -> None:
        for binding in getattr(metadata, "forward_receive_bindings", ()):
            self._install_forward_receive_binding(binding)
        for timeout in getattr(metadata, "decision_timeouts", ()):
            self._control_failed_recving.add(timeout.request_id)
            self._invalid_block_ids.update(timeout.external_block_ids)
        super().start_load_kv(metadata)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_wire_ids = self._release_finished_forward_terminals(finished_req_ids)
        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            assert self.kv_recv_layer_thread is not None
            raw_done = self.kv_recv_layer_thread.get_and_clear_done_requests()
            raw_failed = self.kv_recv_layer_thread.get_and_clear_failed_requests()
        else:
            raw_done = set()
            raw_failed = set()

        ignored_wire_ids = set(self._consumed_forward_terminals)
        ignored_wire_ids.update(
            wire_request_id for wire_request_id in finished_wire_ids if wire_request_id not in self.request_map
        )
        raw_done.difference_update(ignored_wire_ids)
        raw_failed.difference_update(ignored_wire_ids)

        done_wire_ids = raw_done.union(self._pending_forward_done)
        failed_wire_ids = raw_failed.union(self._pending_forward_failed)
        done_wire_ids.difference_update(failed_wire_ids)
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
            first_forward_block = binding.token_start // self.block_size[0]
            last_forward_block = math.ceil(binding.token_end / self.block_size[0])
            self._invalid_block_ids.update(binding.destination_block_ids[0][first_forward_block:last_forward_block])
            forward_finished.add(decode_request_id)
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
            forward_finished.add(decode_request_id)
            self._consume_forward_receive_binding(binding)

        for decode_request_id in ordinary_failed:
            if metadata := self._recving_metadata.get(decode_request_id):
                self._invalid_block_ids.update(block_id for group in metadata.local_block_ids for block_id in group)
        for decode_request_id in ordinary_done.union(ordinary_failed):
            self.request_map.pop(get_external_request_id(decode_request_id), None)
            self._recving_metadata.pop(decode_request_id, None)

        self._pending_forward_done = pending_done
        self._pending_forward_failed = pending_failed
        done_recving = ordinary_done.union(forward_finished, self.virtual_request)
        self.virtual_request = set()
        if done_recving:
            logger.info(
                "Number of completed KV cache recv requests: %s, receive requests: %s",
                len(done_recving),
                done_recving,
            )
        done_recving.update(self._control_failed_recving)
        self._control_failed_recving.clear()
        return set(), done_recving

    def shutdown(self) -> None:
        for binding in self._forward_receive_bindings.values():
            if self.request_map.get(binding.wire_request_id) == binding.decode_request_id:
                self.request_map.pop(binding.wire_request_id)
        self._forward_receive_bindings.clear()
        self._pending_forward_done.clear()
        self._pending_forward_failed.clear()
        self._consumed_forward_terminals.clear()
        self._control_failed_recving.clear()
        if self._kvpool_worker_adapter is not None:
            self._kvpool_worker_adapter.close()


class DualPathConnector(MooncakeLayerwiseConnector, SupportsHMA):
    """Behavior-preserving alias of ``MooncakeLayerwiseConnector``.

    A selectable connector name that constructs DualPath Scheduler/Worker
    subclasses while preserving the parent's accounting, metadata, transfer,
    completion, invalid-block, cleanup, and failure behavior for ordinary
    Layerwise workloads.
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
        return self.connector_worker.get_finished(finished_req_ids)

    def shutdown(self):
        """Release Task-01-owned state and adapters, then defer to the base."""
        if isinstance(self.connector_scheduler, DualPathConnectorScheduler):
            self.connector_scheduler.shutdown()
        if isinstance(self.connector_worker, DualPathConnectorWorker):
            self.connector_worker.shutdown()
        super().shutdown()

# SPDX-License-Identifier: Apache-2.0
"""Worker-side bidirectional transfer lifecycle for DualPath."""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, assert_never

import torch
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.kvpool_adapter import (
    KVPoolWorkerAdapter,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    ForwardReceiveBinding,
    ReversePlan,
    ReverseReceiveBinding,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    PathKind,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorWorker,
    ReqMeta,
    get_external_request_id,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
        AscendConnectorMetadata,
    )


__all__ = ["DualPathConnectorWorker"]


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
    store_load_failed: bool
    terminal_published: bool


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
        self._reverse_terminal_lock = threading.Lock()
        self._pending_local_reverse_terminals: dict[str, bool] = {}
        self._control_failed_recving: set[str] = set()
        self._forward_receive_bindings: dict[str, ForwardReceiveBinding] = {}
        self._pending_forward_done_wire_ids: set[str] = set()
        self._pending_forward_failed_wire_ids: set[str] = set()
        self._consumed_forward_terminal_wire_ids: dict[str, str] = {}
        self._reverse_receive_bindings: dict[str, ReverseReceiveBinding] = {}
        self._reverse_request_map: dict[str, str] = {}
        self._pending_reverse_done_wire_ids: set[str] = set()
        self._pending_reverse_failed_wire_ids: set[str] = set()
        self._consumed_reverse_terminal_wire_ids: dict[str, bool] = {}
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
        # The parent already started the role-native runtime (producer send / consumer
        # receive); this adds the opposite-direction runtime for dual-path transfers.
        if self.dual_path_cfg.role == "prefill":
            self._ensure_receive_layer_runtime()
        else:
            self._ensure_send_layer_runtime()
        if self._kvpool_worker_adapter is not None:
            self._kvpool_worker_adapter.register_kv_caches(kv_caches)

    def _install_forward_receive_binding(self, binding: ForwardReceiveBinding) -> None:
        # PE_READ bindings ride the ordinary Layerwise recv path and own no
        # split state, so they stay accepted after split shutdown; only
        # DE_READ bindings gate the split state machine.
        if binding.path is PathKind.DE_READ and not self._accepting_split_requests:
            return
        consumed_decode_request_id = self._consumed_forward_terminal_wire_ids.get(binding.wire_request_id)
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
            decode_request_id = self._consumed_forward_terminal_wire_ids.get(wire_request_id)
            if decode_request_id in finished_req_ids:
                self._consumed_forward_terminal_wire_ids.pop(wire_request_id)
        self._pending_forward_done_wire_ids.difference_update(finished_wire_ids)
        self._pending_forward_failed_wire_ids.difference_update(finished_wire_ids)
        for request_id in finished_req_ids:
            binding = self._forward_receive_bindings.get(request_id)
            if binding is not None:
                if self.request_map.get(binding.wire_request_id) == request_id:
                    self.request_map.pop(binding.wire_request_id)
                self._forward_receive_bindings.pop(request_id)
        return finished_wire_ids

    def _install_reverse_receive_binding(self, binding: ReverseReceiveBinding) -> None:
        if not self._accepting_split_requests:
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
        if binding.wire_request_id in self._consumed_reverse_terminal_wire_ids:
            if retained_wire_binding == binding:
                return
            raise RuntimeError(
                f"DualPath wire request {binding.wire_request_id} already belongs to a consumed Reverse terminal; "
                "the original binding is preserved"
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
            or (existing_prefill_request_id is not None and existing_prefill_request_id != binding.prefill_request_id)
            or existing_parent_request_id is not None
            or binding.wire_request_id in self._consumed_forward_terminal_wire_ids
            or conflicts_with_retained_binding
        ):
            raise RuntimeError(
                f"DualPath Prefill request {binding.prefill_request_id} got a conflicting duplicate "
                "Reverse receive binding; the original binding is preserved"
            )

        self._reverse_request_map[binding.wire_request_id] = binding.prefill_request_id
        self._reverse_receive_bindings[binding.prefill_request_id] = binding
        if binding.wire_request_id in self._pending_forward_done_wire_ids:
            self._pending_forward_done_wire_ids.remove(binding.wire_request_id)
            self._pending_reverse_done_wire_ids.add(binding.wire_request_id)
        if binding.wire_request_id in self._pending_forward_failed_wire_ids:
            self._pending_forward_failed_wire_ids.remove(binding.wire_request_id)
            self._pending_reverse_failed_wire_ids.add(binding.wire_request_id)

    def _release_split_request_state(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_wire_ids = self._release_finished_forward_terminals(finished_req_ids)
        finished_reverse_wire_ids = self._release_finished_reverse_terminals(finished_req_ids)
        self._control_failed_recving.difference_update(finished_req_ids)
        with self._reverse_terminal_lock:
            for request_id in finished_req_ids:
                self._split_trackers.pop(request_id, None)
                self._pending_local_reverse_terminals.pop(request_id, None)
        return finished_wire_ids, finished_reverse_wire_ids

    def _release_finished_reverse_terminals(self, finished_req_ids: set[str]) -> set[str]:
        finished_wire_ids: set[str] = set()
        for prefill_request_id in finished_req_ids:
            binding = self._reverse_receive_bindings.pop(prefill_request_id, None)
            if binding is None:
                continue
            finished_wire_ids.add(binding.wire_request_id)
            self._reverse_request_map.pop(binding.wire_request_id, None)
            self._consumed_reverse_terminal_wire_ids.pop(binding.wire_request_id, None)
        self._pending_reverse_done_wire_ids.difference_update(finished_wire_ids)
        self._pending_reverse_failed_wire_ids.difference_update(finished_wire_ids)
        return finished_wire_ids

    def _consume_reverse_receive_binding(self, binding: ReverseReceiveBinding, terminal_flag: bool) -> None:
        """Record the terminal and drop the wire-id mapping, but retain the
        binding itself: ``_release_finished_reverse_terminals`` later recovers
        the wire id from it when the Prefill-local request finishes."""
        self._consumed_reverse_terminal_wire_ids[binding.wire_request_id] = terminal_flag
        self._reverse_request_map.pop(binding.wire_request_id, None)

    def _consume_forward_receive_binding(self, binding: ForwardReceiveBinding) -> None:
        """Record the terminal and pop the binding eagerly: unlike the Reverse
        side, the Forward side needs no later lookup because
        ``_release_finished_forward_terminals`` recovers wire ids via
        ``get_external_request_id``."""
        self._consumed_forward_terminal_wire_ids[binding.wire_request_id] = binding.decode_request_id
        self.request_map.pop(binding.wire_request_id, None)
        self._forward_receive_bindings.pop(binding.decode_request_id, None)

    def _install_split_tracker(
        self,
        binding: ForwardReceiveBinding,
        store_metadata: AscendConnectorMetadata | None,
    ) -> None:
        if not self._accepting_split_requests:
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
            store_load_failed=False,
            terminal_published=False,
        )

    def _install_reverse_plan(self, plan: ReversePlan) -> None:
        if not self._accepting_split_requests:
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

        existing_plan = tracker.plan
        conflicts_with_retained_plan = any(
            retained != plan
            and (retained.request_key == plan.request_key or retained.wire_request_id == plan.wire_request_id)
            for retained in (split_tracker.plan for split_tracker in self._split_trackers.values())
            if retained is not None
        )
        if (existing_plan is not None and existing_plan != plan) or conflicts_with_retained_plan:
            raise RuntimeError(
                f"DualPath Decode request {decode_request_id} got a conflicting duplicate Reverse plan; "
                "the original plan is preserved"
            )
        if existing_plan is not None:
            return

        tracker.plan = plan
        tracker.reverse_phase = _SplitPhase.PENDING

    def _build_reverse_send_metadata(
        self,
        plan: ReversePlan,
        decode_request_id: str,
    ) -> MooncakeLayerwiseConnectorMetadata:
        if not (self.pd_head_ratio == 1 and not self.enable_kv_quant and not self.enable_c8_quant):
            raise RuntimeError("DualPath Reverse supports the plain Layerwise send path only")
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

        if len(transfer_mappings) > 1:
            raise RuntimeError(f"Not support add mutil transfer task for req_id:{decode_request_id}")
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
        if not self._accepting_split_requests:
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
        if self._registered_kv_caches is None:
            raise RuntimeError("DualPath Reverse submission requires registered KV caches")
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
        if store_invalid_block_ids:
            for tracker in self._split_trackers.values():
                if tracker.store_phase is _SplitPhase.PENDING and store_invalid_block_ids.intersection(
                    tracker.store_destination_slice
                ):
                    tracker.store_load_failed = True
        for request_id in store_done_recving:
            tracker = self._split_trackers.get(request_id)
            if tracker is None:
                published_store_terminals.add(request_id)
                continue
            if tracker.store_phase is not _SplitPhase.PENDING:
                continue
            if tracker.store_load_failed:
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

    def start_load_kv(self, metadata: DualPathConnectorMetadata) -> None:
        store_metadata = metadata.decode_store_metadata
        is_prefill = self.dual_path_cfg.role == "prefill"
        is_decode = self.dual_path_cfg.role == "decode"
        if is_prefill:
            for binding in metadata.reverse_receive_bindings:
                self._install_reverse_receive_binding(binding)
        de_read_bindings: list[ForwardReceiveBinding] = []
        for binding in metadata.forward_receive_bindings:
            self._install_forward_receive_binding(binding)
            if binding.path is PathKind.DE_READ and is_decode:
                self._install_split_tracker(binding, store_metadata)
                de_read_bindings.append(binding)
        if is_decode:
            for plan in metadata.reverse_plans:
                self._install_reverse_plan(plan)
        for failure in metadata.control_failures:
            self._control_failed_recving.add(failure.request_id)
            self._invalid_block_ids.update(failure.invalid_block_ids)
            logger.warning(
                "dual_path control_terminal key=%s failure_source=%s final_predicate=FAILED",
                failure.request_id,
                failure.reason.value,
            )
        # Non-DE_READ Store loads bypass the split state machine and stay accepted after shutdown.
        split_store_accepted = self._accepting_split_requests or not any(
            binding.path is PathKind.DE_READ for binding in metadata.forward_receive_bindings
        )
        if store_metadata is not None and split_store_accepted:
            assert self._kvpool_worker_adapter is not None
            self._kvpool_worker_adapter.start_load_kv(store_metadata)
        # Reverse submission waits for the plan install loop above; the DE_READ
        # bindings collected in the first pass drive it without re-scanning.
        for binding in de_read_bindings:
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
        metadata: DualPathConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        finished_wire_ids, finished_reverse_wire_ids = self._release_split_request_state(finished_req_ids)

        done_sending: set[str] = set()
        done_recving: set[str] = set()
        store_metadata = metadata.decode_store_metadata
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

        done_recving.update(self._drain_local_reverse_terminals())

        if self.kv_recv_layer_thread is not None:
            raw_done = self.kv_recv_layer_thread.get_and_clear_done_requests()
            raw_failed = self.kv_recv_layer_thread.get_and_clear_failed_requests()
        else:
            raw_done = set()
            raw_failed = set()

        reverse_finished = self._consume_reverse_wire_terminals(
            raw_done, raw_failed, finished_wire_ids, finished_reverse_wire_ids
        )
        pending_done, pending_failed, ordinary_done, ordinary_failed, forward_finished = (
            self._consume_forward_wire_terminals(raw_done, raw_failed)
        )
        self._finish_ordinary_requests(ordinary_done, ordinary_failed)

        self._pending_forward_done_wire_ids = pending_done
        self._pending_forward_failed_wire_ids = pending_failed
        done_recving.update(ordinary_done.union(forward_finished, reverse_finished, self.virtual_request))
        self.virtual_request = set()
        self._log_published_split_terminals(done_recving)
        done_recving.update(self._control_failed_recving)
        self._control_failed_recving.clear()
        return done_sending, done_recving

    def _drain_local_reverse_terminals(self) -> set[str]:
        """Drain locally-observed Reverse send terminals into the split trackers."""
        with self._reverse_terminal_lock:
            local_reverse_terminals = dict(self._pending_local_reverse_terminals)
            self._pending_local_reverse_terminals.clear()
        finished: set[str] = set()
        for request_id, terminal_flag in local_reverse_terminals.items():
            tracker = self._split_trackers.get(request_id)
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
                    finished.add(request_id)
            else:
                tracker.reverse_phase = _SplitPhase.FAILED
                self._invalid_block_ids.update(tracker.forward_destination_slice)
                logger.warning(
                    "dual_path data_terminal key=%s failure_source=REVERSE final_predicate=FAILED",
                    request_id,
                )
                if not tracker.terminal_published:
                    tracker.terminal_published = True
                    finished.add(request_id)
        return finished

    def _consume_reverse_wire_terminals(
        self,
        raw_done: set[str],
        raw_failed: set[str],
        finished_wire_ids: set[str],
        finished_reverse_wire_ids: set[str],
    ) -> set[str]:
        """Attribute wire terminals owned by Reverse bindings to their Prefill
        requests. ``raw_done``/``raw_failed`` are filtered in place: ignored
        and Reverse-owned wire ids are removed before return."""
        ignored_wire_ids = set(self._consumed_forward_terminal_wire_ids).union(self._consumed_reverse_terminal_wire_ids)
        ignored_wire_ids.update(
            wire_request_id for wire_request_id in finished_wire_ids if wire_request_id not in self.request_map
        )
        ignored_wire_ids.update(
            wire_request_id
            for wire_request_id in finished_reverse_wire_ids
            if wire_request_id not in self._reverse_request_map
        )
        raw_done.difference_update(ignored_wire_ids)
        raw_failed.difference_update(ignored_wire_ids)

        reverse_owned_wire_ids = set(self._reverse_request_map)
        reverse_done_wire_ids = raw_done.intersection(reverse_owned_wire_ids).union(self._pending_reverse_done_wire_ids)
        reverse_failed_wire_ids = raw_failed.intersection(reverse_owned_wire_ids).union(
            self._pending_reverse_failed_wire_ids
        )
        raw_done.difference_update(reverse_owned_wire_ids)
        raw_failed.difference_update(reverse_owned_wire_ids)
        reverse_done_wire_ids.difference_update(reverse_failed_wire_ids)
        reverse_finished: set[str] = set()
        for wire_request_id in reverse_failed_wire_ids:
            prefill_request_id = self._reverse_request_map[wire_request_id]
            binding = self._reverse_receive_bindings[prefill_request_id]
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
            prefill_request_id = self._reverse_request_map[wire_request_id]
            binding = self._reverse_receive_bindings[prefill_request_id]
            reverse_finished.add(prefill_request_id)
            self._consume_reverse_receive_binding(binding, True)
            logger.info(
                "dual_path reverse_terminal key=%s terminal=DONE final_predicate=SUCCESS",
                prefill_request_id,
            )
        self._pending_reverse_done_wire_ids.clear()
        self._pending_reverse_failed_wire_ids.clear()
        return reverse_finished

    def _consume_forward_wire_terminals(
        self,
        raw_done: set[str],
        raw_failed: set[str],
    ) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
        """Classify the remaining wire terminals into pending, ordinary, and
        split-Forward buckets. Returns ``(pending_done, pending_failed,
        ordinary_done, ordinary_failed, forward_finished)``."""
        done_wire_ids = raw_done.union(self._pending_forward_done_wire_ids)
        failed_wire_ids = raw_failed.union(self._pending_forward_failed_wire_ids)
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
            if binding.path is PathKind.PE_READ:
                first_forward_block = binding.token_start // self.block_size[0]
                last_forward_block = math.ceil(binding.token_end / self.block_size[0])
                self._invalid_block_ids.update(binding.destination_block_ids[0][first_forward_block:last_forward_block])
                forward_finished.add(decode_request_id)
            elif binding.path is PathKind.DE_READ:
                tracker = self._split_trackers[decode_request_id]
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
            if binding.path is PathKind.PE_READ:
                forward_finished.add(decode_request_id)
            elif binding.path is PathKind.DE_READ:
                tracker = self._split_trackers[decode_request_id]
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
        return pending_done, pending_failed, ordinary_done, ordinary_failed, forward_finished

    def _finish_ordinary_requests(self, ordinary_done: set[str], ordinary_failed: set[str]) -> None:
        """Release ordinary (non-DualPath) recv bookkeeping for finished requests."""
        for decode_request_id in ordinary_failed:
            if meta := self._recving_metadata.get(decode_request_id):
                self._invalid_block_ids.update(block_id for group in meta.local_block_ids for block_id in group)
        for decode_request_id in ordinary_done.union(ordinary_failed):
            self.request_map.pop(get_external_request_id(decode_request_id), None)
            self._recving_metadata.pop(decode_request_id, None)

    def _log_published_split_terminals(self, done_recving: set[str]) -> None:
        """Emit the per-request split summary and the aggregate recv summary."""
        for request_id in done_recving:
            tracker = self._split_trackers.get(request_id)
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

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_block_ids = super().get_block_ids_with_load_errors()
        if self._kvpool_worker_adapter is not None:
            invalid_block_ids.update(self._kvpool_worker_adapter.get_block_ids_with_load_errors())
        return invalid_block_ids

    def shutdown(self) -> None:
        """Release local state without promising cancellation of remote DMA."""
        if not self._accepting_split_requests:
            return
        self._accepting_split_requests = False
        for binding in self._forward_receive_bindings.values():
            if self.request_map.get(binding.wire_request_id) == binding.decode_request_id:
                self.request_map.pop(binding.wire_request_id)
        self._forward_receive_bindings.clear()
        self._pending_forward_done_wire_ids.clear()
        self._pending_forward_failed_wire_ids.clear()
        self._consumed_forward_terminal_wire_ids.clear()
        with self._reverse_terminal_lock:
            self._split_trackers.clear()
            self._pending_local_reverse_terminals.clear()
        self._reverse_receive_bindings.clear()
        self._reverse_request_map.clear()
        self._pending_reverse_done_wire_ids.clear()
        self._pending_reverse_failed_wire_ids.clear()
        self._consumed_reverse_terminal_wire_ids.clear()
        self._control_failed_recving.clear()

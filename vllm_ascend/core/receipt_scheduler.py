# SPDX-License-Identifier: Apache-2.0
"""Receipt target lifetime hooks on vLLM's real free/preemption paths.

The connector's request_finished hook does not cover forced preemption or
failed asynchronous loads. A scheduler subclass is needed for those paths;
there is no model patch or ref-count inflation. Select this class explicitly
for receipt Decode engines, not for ordinary deployments.
"""

import threading
import time
from dataclasses import replace

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import ContractError
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import SerializedKVCacheManager


class ReceiptScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.recompute_kv_load_failures:
            raise ContractError("Receipt Decode requires kv_load_failure_policy='fail'")
        configure = getattr(self.connector, "configure_receipt_scheduler_runtime", None)
        if not callable(configure):
            raise ContractError("ReceiptScheduler requires a receipt-capable target connector")
        lock = threading.RLock()
        manager = SerializedKVCacheManager(self.kv_cache_manager, lock)
        # Refresh connector references installed by the base constructor before
        # starting reverse RPCs. Every child sees the same serialized manager.
        self.connector.bind_kv_cache_manager(manager)
        registry = configure(manager, lock)
        if registry is None:
            raise ContractError("ReceiptScheduler did not bind a real target retention authority")
        self.kv_cache_manager = manager
        self._target_registry = registry
        self._retained_finished: set[str] = set()
        self._target_recv_pending: set[str] = set()

    def _ready_target_receives(self):
        ready = set()
        pending = getattr(self, "_target_recv_pending", set())
        for request_id in tuple(pending):
            request = self.requests.get(request_id)
            if request is None:
                pending.discard(request_id)
            elif request.is_finished():
                ready.add(request_id)
                pending.discard(request_id)
            elif request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                if self.connector.is_target_ready(request):
                    ready.add(request_id)
                    pending.discard(request_id)
            else:
                pending.discard(request_id)  # Stale duplicate from a preempted/running lifetime.
        return ready

    def _update_from_kv_xfer_finished(self, kv_connector_output):
        self._target_recv_pending.update(kv_connector_output.finished_recving or ())
        # Preserve one-shot completions while other TP owners prepare. All
        # other fields (errors, stats, worker metadata) still flow normally.
        filtered = replace(kv_connector_output, finished_recving=self._ready_target_receives())
        super()._update_from_kv_xfer_finished(filtered)

    def _poll_target_receives(self):
        ready = self._ready_target_receives()
        if ready:
            super()._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving=ready))

    def _handle_failed_recving(self, failed_req_ids):
        affected = super()._handle_failed_recving(failed_req_ids)
        # The base only handles WAITING_FOR_REMOTE_KVS. A late contradictory
        # proof must also fail an already-running retained target by request
        # identity, without evicting another pool's overlapping numeric IDs.
        for request_id in failed_req_ids:
            request = self.requests.get(request_id)
            if request is not None and not request.is_finished() and self._target_registry.has_request(request_id):
                affected.add(request_id)
        return affected

    def _target_can_free(self, request) -> bool:
        registry = getattr(self, "_target_registry", None)
        if registry is None or not registry.has_request(request.request_id):
            return True
        return self.connector.prepare_target_free(request)

    def _request_blocks_can_be_freed(self, request) -> bool:
        # This upstream GPU fence is also checked before removing a preempted
        # victim from the running/scheduled lists. Retire only after it passes.
        return super()._request_blocks_can_be_freed(request) and self._target_can_free(request)

    def _free_blocks(self, request):
        if self._target_registry.has_request(request.request_id) and not self._request_blocks_can_be_freed(request):
            self._retained_finished.add(request.request_id)
            return
        super()._free_blocks(request)
        self._retained_finished.discard(request.request_id)

    def _free_request(self, request, delay_free_blocks=False):
        result = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        if request.request_id in self.requests and self._target_registry.has_request(request.request_id):
            self._retained_finished.add(request.request_id)
        return result

    def _drain_retained_finished(self):
        for request_id in tuple(self._retained_finished):
            request = self.requests.get(request_id)
            if request is None:
                self._retained_finished.discard(request_id)
            elif request.is_finished() and self._request_blocks_can_be_freed(request):
                self._free_blocks(request)

    def schedule(self):
        self._poll_target_receives()
        self._drain_retained_finished()
        return super().schedule()

    def has_finished_requests(self):
        return bool(self._retained_finished or self._target_recv_pending) or super().has_finished_requests()

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        if not self._request_blocks_can_be_freed(request):
            raise ContractError("Preemption cannot reset a request with retained target blocks")
        super()._preempt_request(request, timestamp, drop_stale_output=drop_stale_output)

    def _try_promote_blocked_waiting_request(self, request):
        if request.request_id in self.failed_recving_kv_req_ids and not self._request_blocks_can_be_freed(request):
            return False
        return super()._try_promote_blocked_waiting_request(request)

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        if reset_running_requests:
            # The upstream forced-reset path pops victims without its normal
            # preemption preflight. Retire and preempt each victim atomically
            # with respect to this scheduler thread; leave pending victims in
            # their existing queues/counters when a proof is unavailable.
            for request in reversed(tuple(self.running)):
                if not self._request_blocks_can_be_freed(request):
                    return False
                self.running.remove(request)
                self._preempt_request(request, time.monotonic(), drop_stale_output=True)
                self.prev_step_scheduled_req_ids.discard(request.request_id)
        return super().reset_prefix_cache(False, reset_connector)

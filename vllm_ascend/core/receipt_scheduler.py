# SPDX-License-Identifier: Apache-2.0
"""Receipt target lifetime hooks on vLLM's real free/preemption paths.

The connector's request_finished hook does not cover forced preemption or
failed asynchronous loads. A scheduler subclass is needed for those paths;
there is no model patch or ref-count inflation. Select this class explicitly
for receipt Decode engines, not for ordinary deployments.
"""

import threading
import time

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import ContractError
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import SerializedKVCacheManager


class ReceiptScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self._drain_retained_finished()
        return super().schedule()

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

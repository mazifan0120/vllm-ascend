# SPDX-License-Identifier: Apache-2.0
"""Bind layerwise target ownership to vLLM's scheduler and real block lifetime."""

from dataclasses import dataclass, replace
from queue import Empty, SimpleQueue
from time import monotonic

from vllm.logger import logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.core.layerwise_offload_scheduler import DecodeAdmission, DecodeWorkerEvent, DecodeWorkerLink
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_offload_plan import (
    CatalogRef,
    RegisteredLayout,
    TargetAllocation,
    TargetId,
)


@dataclass
class _BoundRequest:
    request: Request
    allocation: TargetAllocation
    retained_at: float
    published_at: float | None = None
    close_started_at: float | None = None
    drained: bool = False

    @property
    def target(self) -> TargetId:
        return self.allocation.target


class LayerwiseDecodeScheduler(Scheduler):
    """Scheduler-thread coordinator for the D engine.

    Keep the original allocation in the cache manager until all owners have
    closed. Holding an extra block-pool reference would turn private blocks
    into shared blocks under vLLM's copy-on-write rules.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layerwise_admission: DecodeAdmission | None = None
        self._layerwise_requests: dict[str, _BoundRequest] = {}
        self._layerwise_targets: dict[TargetId, _BoundRequest] = {}
        self._layerwise_failures: set[TargetId] = set()
        self._layerwise_events: SimpleQueue[DecodeWorkerEvent] = SimpleQueue()
        self._layerwise_free_fence = 0
        self._layerwise_deferred_close: set[TargetId] = set()
        self._layerwise_decode_stats: dict[str, int | float] = {}
        bind = getattr(getattr(self, "connector", None), "bind_layerwise_decode_scheduler", None)
        if callable(bind):
            bind(self)

    def bind_layerwise_workers(self, layout: RegisteredLayout, link: DecodeWorkerLink) -> DecodeAdmission:
        """Called once after registration. Worker events must enter on the scheduler thread."""
        if self._layerwise_admission is not None:
            raise RuntimeError("Layerwise workers already bound")
        if self.recompute_kv_load_failures:
            raise ValueError("Layerwise D Host targets require kv_load_failure_policy='fail'")
        self._layerwise_link = link
        self._layerwise_admission = DecodeAdmission(layout, self._assert_layerwise_blocks, self._layerwise_closed)
        return self._layerwise_admission

    def retain_layerwise_target(self, request: Request, allocation: TargetAllocation) -> tuple[CatalogRef, ...]:
        admission = self._layerwise_admission
        if admission is None or self.requests.get(request.request_id) is not request or request.is_finished():
            raise RuntimeError("Target requires a live request on a bound layerwise scheduler")
        target = allocation.target
        previous = self._layerwise_requests.get(request.request_id)
        if previous is not None:
            if previous.target != target or previous.request is not request or previous.drained:
                raise RuntimeError("Cannot replace a retained layerwise target")
            return admission.retain(allocation)
        if target in self._layerwise_targets:
            raise RuntimeError("Target is already associated with another engine request")
        binding = _BoundRequest(request, allocation, monotonic())
        self._layerwise_requests[request.request_id] = binding
        self._layerwise_targets[target] = binding
        try:
            refs = admission.retain(allocation)
        except Exception:
            del self._layerwise_requests[request.request_id]
            del self._layerwise_targets[target]
            raise
        self._layerwise_link.prepare(allocation, refs)
        return refs

    def post_layerwise_event(self, event: DecodeWorkerEvent) -> None:
        """Called by the control loop, which may be on another thread."""
        self._layerwise_events.put(event)

    def take_layerwise_decode_stats(self) -> dict[str, int | float]:
        stats, self._layerwise_decode_stats = self._layerwise_decode_stats, {}
        return stats

    def _record_layerwise_phase(self, phase: str, seconds: float) -> None:
        stats = self._layerwise_decode_stats
        stats[phase] = stats.get(phase, 0) + 1
        key = f"{phase}_seconds"
        stats[key] = stats.get(key, 0) + seconds

    def schedule(self, *args, **kwargs):
        self._drain_layerwise_events()
        for target in tuple(self._layerwise_deferred_close):
            self._begin_layerwise_close(self._layerwise_targets[target])
        return super().schedule(*args, **kwargs)

    def layerwise_scheduled_targets(self, output) -> tuple[TargetId, ...]:
        if not self._layerwise_requests:
            return ()
        new_blocks = {request.req_id: request.block_ids for request in output.scheduled_new_reqs}
        resumed = output.scheduled_cached_reqs.resumed_req_ids
        targets = []
        for request_id in output.num_scheduled_tokens:
            binding = self._layerwise_requests.get(request_id)
            if binding is None:
                continue
            if request_id in resumed:
                raise RuntimeError("A retained Host target cannot resume against a replaced allocation")
            if (blocks := new_blocks.get(request_id)) is not None:
                admitted = binding.allocation.blocks
                if len(blocks) != len(admitted) or any(
                    tuple(group[: len(expected)]) != expected for group, expected in zip(blocks, admitted, strict=True)
                ):
                    raise RuntimeError("Decode was scheduled against a different Host target allocation")
            targets.append(binding.target)
        return tuple(targets)

    def _drain_layerwise_events(self) -> None:
        admission = self._layerwise_admission
        if admission is None:
            return
        while True:
            try:
                event = self._layerwise_events.get_nowait()
            except Empty:
                return
            target, worker = event.target, event.worker
            if event.kind == "prepared":
                admission.prepared(target, worker)
                if target not in self._layerwise_failures and (published := admission.publish(target)) is not None:
                    binding = self._layerwise_targets[target]
                    binding.published_at = monotonic()
                    self._record_layerwise_phase("prepare", binding.published_at - binding.retained_at)
                    self._layerwise_link.publish(binding.request, *published)
            elif event.kind == "ready":
                if admission.ready(target, worker):
                    binding = self._layerwise_targets[target]
                    if binding.published_at is not None:
                        self._record_layerwise_phase("ready", monotonic() - binding.published_at)
                    logger.info(
                        "Layerwise Host target ready request=%s tp=%d",
                        binding.request.request_id,
                        admission.worker_count,
                    )
            elif event.kind == "closed":
                admission.closed(target, worker)
            elif event.kind == "failed":
                admission.validate_worker(worker)
                self.layerwise_target_failed(target)
            else:
                raise ValueError("Unknown layerwise worker event")

    def _assert_layerwise_blocks(self, allocation: TargetAllocation) -> None:
        request = self._layerwise_targets[allocation.target].request
        groups = self.kv_cache_manager.get_blocks(request.request_id).blocks
        if request.num_prompt_tokens != allocation.prompt_tokens or len(groups) != len(allocation.blocks):
            raise ValueError("Layerwise target does not match the allocated prompt")
        for live, ids in zip(groups, allocation.blocks, strict=True):
            blocks = live[: len(ids)]
            if tuple(block.block_id for block in blocks) != ids:
                raise ValueError("Layerwise target does not match the real scheduler block IDs")
            if any(block.is_null or block.ref_cnt != 1 or block.block_hash is not None for block in blocks):
                raise ValueError("Remote writes require private, uncached destination blocks")

    def layerwise_target_failed(self, target: TargetId) -> None:
        """Fail through the normal scheduler output path, including while already decoding."""
        binding = self._layerwise_targets.get(target)
        if binding is not None and not binding.request.is_finished():
            self._layerwise_failures.add(target)

    def _layerwise_closed(self, allocation: TargetAllocation) -> None:
        binding = self._layerwise_targets[allocation.target]
        if binding.close_started_at is not None:
            self._record_layerwise_phase("close", monotonic() - binding.close_started_at)
        binding.drained = True
        assert self._layerwise_admission is not None
        logger.info(
            "Layerwise Host target closed request=%s tp=%d",
            binding.request.request_id,
            self._layerwise_admission.worker_count,
        )
        if binding.request.is_finished():
            self._free_blocks(binding.request)

    def _begin_layerwise_close(self, binding: _BoundRequest) -> None:
        # A scheduled batch can reach the worker after a side-channel Close.
        # Let that batch bind first, then let the worker fence its actual accesses.
        if not super()._request_blocks_can_be_freed(binding.request):
            self._layerwise_deferred_close.add(binding.target)
            return
        self._layerwise_deferred_close.discard(binding.target)
        admission = self._layerwise_admission
        assert admission is not None
        ranks = admission.begin_close(binding.target)
        if ranks:
            if binding.close_started_at is None:
                binding.close_started_at = monotonic()
                stats = self._layerwise_decode_stats
                stats["close_started"] = stats.get("close_started", 0) + 1
            self._layerwise_link.close(binding.target, ranks)

    def _connector_finished(self, request):
        if request.request_id in self._layerwise_requests:
            # The stock connector's finish hook can prune blocks before it
            # calls request_finished. Use the layerwise close handshake here.
            return True, None
        return super()._connector_finished(request)

    def _free_request(self, request, delay_free_blocks=False):
        binding = self._layerwise_requests.get(request.request_id)
        output = super()._free_request(request, delay_free_blocks or binding is not None)
        if binding is not None:
            if binding.drained:
                self._free_blocks(request)
            else:
                self._begin_layerwise_close(binding)
        return output

    def _free_blocks(self, request):
        binding = self._layerwise_requests.get(request.request_id)
        if binding is None:
            return super()._free_blocks(request)
        if binding.request is not request or not binding.drained:
            return
        if not super()._request_blocks_can_be_freed(request):
            self._layerwise_free_fence = max(self._layerwise_free_fence, self.sched_step_seq)
        super()._free_blocks(request)
        del self._layerwise_requests[request.request_id]
        del self._layerwise_targets[binding.target]
        self._layerwise_failures.discard(binding.target)
        self._layerwise_deferred_close.discard(binding.target)

    def _request_blocks_can_be_freed(self, request):
        binding = self._layerwise_requests.get(request.request_id)
        if binding is not None and not binding.drained:
            # The stock scheduler asks this before choosing a preemption
            # victim. Fail the D request and close it without HBM recompute.
            self.layerwise_target_failed(binding.target)
            return False
        return super()._request_blocks_can_be_freed(request)

    def _free_request_blocks(self, request):
        binding = self._layerwise_requests.get(request.request_id)
        if binding is not None and not binding.drained:
            raise RuntimeError("Layerwise blocks reached the allocator before target close")
        super()._free_request_blocks(request)

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        if self._layerwise_requests or self._layerwise_free_fence > self.processed_step_seq:
            if reset_running_requests:
                self._layerwise_failures.update(self._layerwise_targets)
            return False  # Retry when the tracked targets are safely closed.
        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def _try_promote_blocked_waiting_request(self, request):
        binding = self._layerwise_requests.get(request.request_id)
        if binding is not None and request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            admission = self._layerwise_admission
            assert admission is not None
            if request.request_id in self.failed_recving_kv_req_ids:
                self.layerwise_target_failed(binding.target)
                return False
            if binding.target in self._layerwise_failures or not admission.may_decode(binding.target):
                return False
            self.finished_recving_kv_req_ids.add(request.request_id)
        return super()._try_promote_blocked_waiting_request(request)

    def _update_from_kv_xfer_finished(self, kv_connector_output):
        if self._layerwise_requests:
            # Layerwise readiness and release use their own rank fan-in.
            kv_connector_output = replace(
                kv_connector_output,
                finished_recving=(kv_connector_output.finished_recving or set()) - self._layerwise_requests.keys(),
                finished_sending=(kv_connector_output.finished_sending or set()) - self._layerwise_requests.keys(),
            )
        return super()._update_from_kv_xfer_finished(kv_connector_output)

    def _handle_failed_recving(self, failed_req_ids):
        failed = super()._handle_failed_recving(failed_req_ids)
        return failed | (failed_req_ids & self._layerwise_requests.keys())

    def update_from_output(self, scheduler_output, model_runner_output):
        if self._layerwise_failures:
            failed = {
                binding.request.request_id
                for target in self._layerwise_failures
                if (binding := self._layerwise_targets.get(target)) is not None and not binding.request.is_finished()
            }
            self._layerwise_failures.clear()
            if failed:
                output = model_runner_output.kv_connector_output or KVConnectorOutput()
                output = replace(output, failed_recving=failed | (output.failed_recving or set()))
                model_runner_output = replace(model_runner_output, kv_connector_output=output)
        return super().update_from_output(scheduler_output, model_runner_output)


class AsyncLayerwiseDecodeScheduler(LayerwiseDecodeScheduler, AsyncScheduler):
    """Use the upstream async hooks with the same layerwise block lifetime."""

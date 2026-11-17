# SPDX-License-Identifier: Apache-2.0
"""Finish internal P/D requests after a connector-only Prefill step."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import RequestStatus


class PrefillRestoreScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.recompute_kv_load_failures or self.vllm_config.use_v2_model_runner:
            raise ValueError("Prefill restore requires the V1 runner and kv_load_failure_policy='fail'")
        configure = getattr(self.connector, "configure_prefill_restore", None)
        if not callable(configure):
            raise ValueError("Prefill restore requires AscendMultiConnector")
        configure()

    @staticmethod
    def _restore_requests(metadata) -> set[str]:
        return {
            request_id
            for child in getattr(metadata, "metadata", ())
            for request_id in getattr(child, "prefill_restores", ())
        }

    def _build_kv_connector_meta(self, connector, scheduler_output):
        metadata = super()._build_kv_connector_meta(connector, scheduler_output)
        covered = {
            request.req_id
            for child in metadata.metadata
            for request in getattr(child, "requests", ())
            if getattr(request, "load_spec", None) is not None
            and request.load_spec.restore_only
            and request.prefill_history == request.target_token_len
        }
        if covered != self._restore_requests(metadata):
            raise RuntimeError("Prefill restore Store and sender plans disagree")
        return metadata

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        kv_output = model_runner_output.kv_connector_output
        if kv_output is None or not kv_output.finished_recving:
            return outputs

        restored = self._restore_requests(scheduler_output.kv_connector_metadata)
        # The base scheduler has processed transfer failures and all-rank completion.
        ready = restored & kv_output.finished_recving & self.finished_recving_kv_req_ids
        ready.difference_update(kv_output.failed_recving or ())
        for request in self.finish_requests(ready, RequestStatus.FINISHED_STOPPED):
            client = request.client_index
            output = outputs.setdefault(client, EngineCoreOutputs())
            output.outputs.append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                    prefill_stats=request.take_prefill_stats(),
                )
            )
            if self.finished_req_ids_dict is not None:
                finished = self.finished_req_ids_dict.pop(client, set())
                output.finished_requests = (output.finished_requests or set()) | finished
        return outputs


class AsyncPrefillRestoreScheduler(PrefillRestoreScheduler, AsyncScheduler):
    """Allow the synchronous Store transfer to share a step with asynchronous scheduling."""

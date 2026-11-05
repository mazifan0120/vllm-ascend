# SPDX-License-Identifier: Apache-2.0
"""SourceReady-driven native writes on the existing per-rank send thread.

Preparing a job is local and nonblocking. Its source reader is registered by
the consumer before publish() queues it. Native completion releases that reader
before the independent proof thread waits on peer acknowledgements. Unknown
native failures never certify source quiescence or target retirement.
"""

import logging
import queue
import threading
from dataclasses import dataclass, field

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    Status,
    TargetClosedError,
    TransferResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import ManifestWritePlan, plan_layer_writes
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_receipt import _ranges_cover


@dataclass
class _RequestWrites:
    request_id: str
    metadata: object
    plan: ManifestWritePlan
    required: dict
    ranges: dict = field(default_factory=dict)
    pending: int = 0
    closed: bool = False
    failed: bool = False
    cancelled: bool = False
    unknown: bool = False
    terminal_sent: bool = False
    addresses: tuple | None = None
    delivery_error: Exception | None = None
    finished: bool = False
    prepared_ranges: dict = field(default_factory=dict)


class SourceWriteJob:
    def __init__(self, writer, state, source, request, ready_event) -> None:
        self._writer = writer
        self.state = state
        self.source = source
        self.request = request
        self.ready_event = ready_event
        self._completion = None
        self._published = False
        self._executed = False

    def publish(self, completion) -> None:
        if self._published or not callable(completion):
            raise ContractError("Publish each registered source read job exactly once")
        self._completion = completion
        self._published = True
        try:
            self._writer.send_queue.put(self)
        except Exception as error:
            self._writer._finish_job(self, (), error, True)
            raise

    def execute(self, engine) -> None:
        if not self._published or self._executed:
            raise ContractError("Source write job was not admitted or already executed")
        self._executed = True
        self._writer._execute(self, engine)


class ManifestSourceWriter:
    def __init__(self, producer_rank, send_queue, resolve_addresses, send_proofs) -> None:
        if (
            type(producer_rank) is not int
            or producer_rank < 0
            or not all(map(callable, (resolve_addresses, send_proofs)))
        ):
            raise ContractError("Manifest writer requires an explicit producer and peer transports")
        self.rank = producer_rank
        self.send_queue = send_queue
        self._resolve = resolve_addresses
        self._send_proofs = send_proofs
        self._lock = threading.RLock()
        self._requests = {}
        # Compact lifetime tombstones survive heavy plan cleanup. A repeated
        # HTTP notification must not reopen a target after its terminal closed
        # all future writers, even if vLLM assigns another internal request ID.
        self._target_owners = {}
        self._unquiesced = []
        self._closing = False
        self._closed = False
        self._proof_queue = queue.Queue()
        self._logger = logging.getLogger(__name__)
        self._proof_thread = threading.Thread(target=self._run_proofs, name="LayerwiseWriteProofs", daemon=True)
        self._proof_thread.start()

    def register_request(self, request_id, metadata) -> None:
        manifest = metadata.remote_target_manifest
        if manifest is None or metadata.remote_host is None or metadata.remote_port is None:
            raise ContractError("Direct writes require an admitted manifest and receiver side channel")
        if manifest.lease.request.request_instance != metadata.request_instance or self.rank >= len(manifest.targets):
            raise ContractError("Producer request/rank does not match the admitted manifest")
        with self._lock:
            if self._closing:
                raise ContractError("Cannot register sources after writer shutdown")
            previous = self._requests.get(request_id)
            target = manifest.lease.request
            owner = self._target_owners.get(target)
            if owner is not None and (owner != request_id or previous is None):
                raise ContractError("Producer target lifetime was already assigned or closed")
            if previous is not None:
                if previous.plan.manifest != manifest or (
                    previous.metadata.remote_host,
                    previous.metadata.remote_port,
                ) != (metadata.remote_host, metadata.remote_port):
                    raise ContractError("Producer request target changed within one admitted lifetime")
                return
            plan = ManifestWritePlan(manifest)
            required = {r.key.operation_id: r.key for r in plan.requirements if r.key.rank == self.rank}
            if not required:
                raise ContractError("Producer has no admitted component entries")
            self._requests[request_id] = _RequestWrites(request_id, metadata, plan, required)
            self._target_owners[target] = request_id

    def prepare_job(self, source, request, ready_event) -> SourceWriteJob:
        with self._lock:
            state = self._requests.get(request.request_id)
            if self._closing or state is None or state.closed or state.finished:
                raise ContractError("Source read belongs to an unknown or closed producer request")
            if not callable(getattr(ready_event, "synchronize", None)):
                raise ContractError("Source write requires the actual recorded source-ready fence")
            state.pending += 1
            for requirement in state.plan.for_layer(self.rank, source.layer):
                state.prepared_ranges.setdefault(requirement.key.operation_id, set()).add(
                    (request.start_token, request.end_token)
                )
            return SourceWriteJob(self, state, source, request, ready_event)

    def finish_requests(self, request_ids) -> None:
        """Worker-confirmed end forbids future jobs, but queued reads still drain.

        A complete prepared plan can finish normally after model return. A
        partial plan (e.g. cancellation before all layers) closes as failed,
        never as a successful empty transfer or an artificial DMA fence.
        """
        proofs = []
        with self._lock:
            for request_id in request_ids:
                state = self._requests.get(request_id)
                if state is None:
                    continue
                state.finished = True
                if not all(
                    _ranges_cover(state.prepared_ranges.get(op, ()), key.token_range.end)
                    for op, key in state.required.items()
                ):
                    state.cancelled = state.closed = True
                if state.closed and not state.pending and not state.terminal_sent:
                    state.terminal_sent = True
                    verdict = Status.FAILED if state.failed else Status.CANCELLED
                    proofs.append((state, (), (verdict, not state.unknown)))
                if state.closed and not state.pending and not state.unknown:
                    self._requests.pop(request_id)
        for proof in proofs:
            self._proof_queue.put(proof)

    def _execute(self, job, engine) -> None:
        with self._lock:
            cancelled = job.state.cancelled and not job.state.failed
        if cancelled:
            self._finish_job(job, (), None, True)
            return  # No read occurred; cancellation does not poison the source epoch.
        results = []
        native_started = False
        quiesced = True
        error = None
        try:
            with self._lock:
                if job.state.failed:
                    raise ContractError("Producer request closed after an earlier write failure")
            # Both address lookup and geometry planning are control work, not
            # source reads. No native submission may precede this real fence.
            job.ready_event.synchronize()
            if job.state.addresses is None:
                job.state.addresses = self._resolve(job.state.metadata, job.state.plan, self.rank)
            plans = plan_layer_writes(
                job.source, job.request, job.state.plan, producer_rank=self.rank, addresses=job.state.addresses
            )
            groups = {}
            for plan in plans:
                if plan.key.token_range.start == plan.key.token_range.end:
                    continue
                groups.setdefault((plan.endpoint, plan.memory_kind), []).append(plan)
            for (endpoint, _), writes in groups.items():
                sources = [address for write in writes for address in write.source_addresses]
                destinations = [address for write in writes for address in write.target_addresses]
                lengths = [length for write in writes for length in write.lengths]
                if lengths:
                    native_started = True
                    quiesced = False
                    result = engine.batch_transfer_sync_write(endpoint, sources, destinations, lengths)
                    if type(result) is not int or result != 0:
                        raise ContractError(f"Native component WRITE did not complete successfully: {result!r}")
                    quiesced = True
                    native_started = False
                results.extend(
                    (TransferResult(write.key, Status.SUCCESS, True), write.receiver_ranks) for write in writes
                )
        except Exception as caught:
            if isinstance(caught, TargetClosedError) and not native_started:
                with self._lock:
                    job.state.cancelled = job.state.closed = True
                self._finish_job(job, tuple(results), None, True)
                return
            error = caught
            # A failed fence/lookup/plan has submitted no source reader. A
            # failing or raising native call has an unknown DMA outcome.
            if native_started:
                quiesced = False
        self._finish_job(job, tuple(results), error, quiesced)

    def _finish_job(self, job, results, error, quiesced) -> None:
        terminal = None
        with self._lock:
            state = job.state
            state.pending -= 1
            if state.pending < 0:
                raise ContractError("Producer source job completed more than once")
            if error is not None:
                state.failed = True
                state.closed = True
            if not quiesced:
                state.unknown = True
                self._unquiesced.append(job)
            for result, _ in results:
                span = result.key.token_range
                state.ranges.setdefault(result.key.operation_id, set()).add((span.start, span.end))
            covered = all(
                _ranges_cover(state.ranges.get(op, ()), key.token_range.end) for op, key in state.required.items()
            )
            if covered:
                state.closed = True
            if state.closed and not state.pending and not state.terminal_sent:
                state.terminal_sent = True
                verdict = Status.FAILED if state.failed else Status.CANCELLED if state.cancelled else Status.SUCCESS
                terminal = (verdict, not state.unknown)
            if state.finished and state.closed and not state.pending and not state.unknown:
                self._requests.pop(state.request_id, None)
        # The exact epoch read ends on native completion. Peer ACK/D decode
        # must never keep the physical P slot busy (design I4).
        job._completion(error, quiesced)
        if results or terminal is not None:
            self._proof_queue.put((state, results, terminal))

    def _run_proofs(self) -> None:
        while True:
            item = self._proof_queue.get()
            if item is None:
                close = getattr(self._send_proofs, "close", None)
                if callable(close):
                    close()
                self._proof_queue.task_done()
                return
            state, results, terminal = item
            try:
                # One exhausted peer operation fails this request's delivery.
                # Do not spend the same timeout again for every queued layer;
                # keep D fail-closed until the separate retirement protocol.
                if state.delivery_error is None:
                    self._send_proofs(state.metadata, results, terminal)
            except Exception as error:
                with self._lock:
                    state.delivery_error = error
                self._logger.error("Manifest completion proof delivery failed; D must retain target: %s", error)
            finally:
                self._proof_queue.task_done()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._unquiesced or any(s.pending for s in self._requests.values()):
                raise ContractError("Source writer still has pending or unknown native reads")
            if not self._closing:
                self._closing = True
                self._proof_queue.put(None)
        self._proof_thread.join(timeout=1)
        if self._proof_thread.is_alive():
            raise ContractError("Source proof delivery is still running")
        self._closed = True

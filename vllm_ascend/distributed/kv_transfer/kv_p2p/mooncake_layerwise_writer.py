# SPDX-License-Identifier: Apache-2.0
"""SourceReady-driven native writes on the existing per-rank send thread.

Preparing a job is local and nonblocking. Its source reader is registered by
the consumer before publish() queues it. Native completion releases that reader
before the independent proof thread waits on peer acknowledgements. Unknown
native failures never certify source quiescence or target retirement.
"""

import queue
import threading
import time
from dataclasses import dataclass, field

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    Status,
    TargetClosedError,
    TransferResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import ManifestWritePlan, plan_layer_writes
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_receipt import _ranges_cover

SOURCE_WRITER_SHUTDOWN_TIMEOUT_SECONDS = 5.0


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
    first_error: Exception | None = None
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
        self._drained = threading.Condition(self._lock)
        self._callbacks_in_progress = 0
        self._requests = {}
        # Compact lifetime tombstones survive heavy plan cleanup. A repeated
        # HTTP notification must not reopen a target after its terminal closed
        # all future writers, even if vLLM assigns another internal request ID.
        self._target_owners = {}
        self._unquiesced = []
        self._closing = False
        self._closed = False
        self._proof_stop_queued = False
        self._proof_queue = queue.Queue()
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

    def prepare_job(self, source, request, ready_event) -> SourceWriteJob | None:
        with self._lock:
            state = self._requests.get(request.request_id)
            if self._closing or state is None:
                raise ContractError("Source read belongs to an unknown or closed producer request")
            if state.failed:
                raise ContractError("Producer request closed after an earlier write failure") from state.first_error
            if state.cancelled:
                # The P forward may still reach later layers after D closes
                # admission. Those layers must not register new DMA readers.
                return None
            requirements = state.plan.for_layer(self.rank, source.layer)
            # An Indexer-only rank may finish its last write before the model
            # reaches later layers whose Main data is sent by rank zero.
            if not requirements:
                return None
            if state.closed or state.finished:
                raise ContractError("Source read belongs to an unknown or closed producer request")
            if not callable(getattr(ready_event, "synchronize", None)):
                raise ContractError("Source write requires the actual recorded source-ready fence")
            state.pending += 1
            for requirement in requirements:
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
                    self._proof_queue.put((state, (), (verdict, not state.unknown)))
                if state.closed and not state.pending and not state.unknown:
                    self._requests.pop(request_id)

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
                    with self._lock:
                        if job.state.cancelled and not job.state.failed:
                            break
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
                if state.first_error is None:
                    state.first_error = error
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
            self._callbacks_in_progress += 1
        # The exact epoch read ends on native completion. Peer ACK/D decode
        # must never keep the physical P slot busy (design I4).
        try:
            job._completion(error, quiesced)
            if results or terminal is not None:
                self._proof_queue.put((state, results, terminal))
        finally:
            with self._drained:
                self._callbacks_in_progress -= 1
                self._drained.notify_all()

    def _cancel_from_peer(self, state) -> tuple[Status, bool] | None:
        with self._lock:
            if state.cancelled or state.terminal_sent:
                return None
            state.cancelled = state.closed = True
            logger.info("Peer closed target for producer request %s", state.request_id)
            if state.pending:
                return None
            state.terminal_sent = True
            verdict = Status.FAILED if state.failed else Status.CANCELLED
            return verdict, not state.unknown

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
                    if self._send_proofs(state.metadata, results, terminal):
                        cancelled = self._cancel_from_peer(state)
                        if cancelled is not None:
                            # Complete this proof before the queued shutdown
                            # sentinel can stop the transport.
                            self._send_proofs(state.metadata, (), cancelled)
            except Exception as error:
                with self._lock:
                    state.delivery_error = error
                logger.error("Manifest completion proof delivery failed; D must retain target: %s", error)
            finally:
                self._proof_queue.task_done()

    def close(self) -> None:
        deadline = time.monotonic() + SOURCE_WRITER_SHUTDOWN_TIMEOUT_SECONDS
        with self._drained:
            if self._closed:
                return
            if not self._closing:
                self._closing = True
                self.finish_requests(tuple(self._requests))
            drained = self._drained.wait_for(
                lambda: bool(self._unquiesced)
                or (not self._callbacks_in_progress and not any(state.pending for state in self._requests.values())),
                timeout=max(0.0, deadline - time.monotonic()),
            )
            if not drained or self._unquiesced:
                raise ContractError("Source writer still has pending or unknown native reads")
            if not self._proof_stop_queued:
                self._proof_stop_queued = True
                self._proof_queue.put(None)
        self._proof_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._proof_thread.is_alive():
            raise ContractError("Source proof delivery is still running")
        with self._lock:
            self._closed = True

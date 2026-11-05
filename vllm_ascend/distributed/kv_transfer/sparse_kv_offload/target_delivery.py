# SPDX-License-Identifier: Apache-2.0
"""Worker-local admission before the scheduler can notify the producer.

Every owner installs the exact manifest/ledger and request mapping before its
acknowledgement. Fast Store restores therefore cannot outrun scheduler metadata
delivery. No registration address is accepted or returned by these controls.
"""

import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    TargetClosedError,
    TargetLease,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_exact_receipt import ExactReceiptLedger
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import ManifestWritePlan
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_wire import lease_fingerprint
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_manifest import TargetManifest


@dataclass
class _Delivery:
    manifest: TargetManifest
    request_id: str
    lease: TargetLease
    ledger: ExactReceiptLedger
    installed: bool = False
    lookup_issued: bool = False
    aborted: bool = False
    releasing: bool = False


class WorkerTargetDelivery:
    def __init__(self, rank, provider, install, *, retire=None) -> None:
        if type(rank) is not int or rank < 0 or not callable(install):
            raise ContractError("Target delivery requires an explicit owner rank and receive installer")
        self._rank = rank
        self._provider = provider
        self._install = install
        self._lock = threading.RLock()
        self._deliveries = {}
        self._by_request_id = {}
        self._retired = {}
        self._retire = retire

    def admit_target(self, manifest: TargetManifest, request_id: str) -> None:
        if type(manifest) is not TargetManifest or type(request_id) is not str or not request_id:
            raise ContractError("Target admission requires a validated manifest and internal request mapping")
        lease = manifest.for_rank(self._rank).lease
        with self._lock:
            previous = self._deliveries.get(lease.request)
            if previous is not None:
                if previous.manifest != manifest or previous.request_id != request_id or not previous.installed:
                    raise ContractError("Target admission is changed or incompletely installed")
                return
            if request_id in self._by_request_id:
                old = self._by_request_id[request_id]
                if old not in self._retired:
                    raise ContractError("A live target request ID cannot change allocation generation")
            if lease.request in self._retired:
                raise TargetClosedError("Target delivery was already retired")
            if self._provider.get_target_views(lease) != lease.views:
                raise ContractError("Target admission does not match this registered owner's live lease")
            plan = ManifestWritePlan(manifest)
            ledger = ExactReceiptLedger(
                lease.request,
                plan.receive_keys(self._rank),
                expected_ranks=tuple(range(len(manifest.targets))),
                target_generation=lease.allocation.allocation_generation,
                visibility_rank=self._rank,
            )
            delivery = _Delivery(manifest, request_id, lease, ledger)
            self._deliveries[lease.request] = delivery
            self._by_request_id[request_id] = lease.request
            # This installer is a worker-local dictionary update, with no
            # peer calls or waits. Uncertain installation remains recorded.
            self._install(request_id, lease, ledger)
            delivery.installed = True

    @contextmanager
    def use_context(self, request_ids):
        with self._lock:
            leases = []
            for request_id in request_ids:
                key = self._by_request_id.get(request_id)
                if key is None:
                    continue  # Ordinary local requests have no remote target lease.
                delivery = self._deliveries.get(key)
                if (
                    delivery is None
                    or delivery.releasing
                    or not delivery.installed
                    or not delivery.ledger.visibility_installed
                ):
                    raise ContractError("Model execution reached an unready remote target")
                leases.append(delivery.lease)
        # Runtime.use can validate reverse retention and record native events;
        # neither it nor model execution may run under the delivery state lock.
        with ExitStack() as stack:
            for lease in leases:
                stack.enter_context(self._provider.use(lease))
            yield

    @contextmanager
    def source_lookup(self, lease):
        with self._lock:
            delivery = self._deliveries.get(lease.request)
            if delivery is None:
                if self._retired.get(lease.request) == lease_fingerprint(lease):
                    raise TargetClosedError("Target delivery has retired")
                raise ContractError("Source lookup has no admitted receive delivery")
            if delivery.lease != lease or not delivery.installed:
                raise ContractError("Source lookup changed the admitted owner lease")
            if delivery.releasing or not delivery.ledger.accepts_source_lookup(self._rank):
                raise TargetClosedError("Target writer admission is closed")
            delivery.lookup_issued = True
            # The owner resolves only local addresses under its inspection
            # lock; no RPC/native wait occurs in this critical section.
            yield

    def abort_requests(self, request_ids):
        with self._lock:
            for request_id in request_ids:
                key = self._by_request_id.get(request_id)
                delivery = self._deliveries.get(key)
                if delivery is not None:
                    delivery.aborted = True
                    delivery.ledger.abort_delivery()

    def release_target(self, lease):
        with self._lock:
            retired = self._retired.get(lease.request)
            if retired is not None:
                if retired != lease_fingerprint(lease):
                    raise ContractError("Retired delivery lease changed")
                return
            delivery = self._deliveries.get(lease.request)
            if delivery is not None:
                if delivery.lease != lease or delivery.releasing:
                    raise ContractError("Target delivery changed or is already releasing")
                if delivery.installed and not (
                    delivery.ledger.remote_quiesced()
                    or (delivery.aborted or delivery.ledger.delivery_aborted)
                    and not delivery.lookup_issued
                ):
                    raise ContractError("Target still has possible remote native readers")
                delivery.releasing = True
        try:
            self._provider.release_target(lease)
        except Exception:
            with self._lock:
                if delivery is not None:
                    delivery.releasing = False
            raise
        with self._lock:
            self._retired[lease.request] = lease_fingerprint(lease)
            self._deliveries.pop(lease.request, None)
        if delivery is not None and callable(self._retire):
            self._retire(delivery.request_id, lease, delivery.ledger)

    def delivery_status(self, lease: TargetLease) -> dict:
        with self._lock:
            delivery = self._deliveries.get(lease.request)
            if delivery is None:
                if self._retired.get(lease.request) == lease_fingerprint(lease):
                    return {"ready": False, "remote_quiesced": True}
                # No admission => no peer lookup can expose these regions.
                self._provider.get_target_views(lease)
                return {"ready": False, "remote_quiesced": True}
            if delivery.lease != lease:
                raise ContractError("Unknown or incompletely admitted target delivery")
            if not delivery.installed:
                return {"ready": False, "remote_quiesced": True}  # No source gate ever opened.
            return {
                "ready": not delivery.releasing and delivery.ledger.should_publish(),
                "remote_quiesced": delivery.ledger.remote_quiesced()
                or (delivery.aborted or delivery.ledger.delivery_aborted)
                and not delivery.lookup_issued,
            }

# SPDX-License-Identifier: Apache-2.0
"""Worker-local admission before the scheduler can notify the producer.

Every owner installs the exact manifest/ledger and request mapping before its
acknowledgement. Fast Store restores therefore cannot outrun scheduler metadata
delivery. No registration address is accepted or returned by these controls.
"""

import threading
from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import ContractError, TargetLease
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_exact_receipt import ExactReceiptLedger
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import ManifestWritePlan
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_manifest import TargetManifest


@dataclass
class _Delivery:
    manifest: TargetManifest
    request_id: str
    lease: TargetLease
    ledger: ExactReceiptLedger
    installed: bool = False


class WorkerTargetDelivery:
    def __init__(self, rank, provider, install) -> None:
        if type(rank) is not int or rank < 0 or not callable(install):
            raise ContractError("Target delivery requires an explicit owner rank and receive installer")
        self._rank = rank
        self._provider = provider
        self._install = install
        self._lock = threading.RLock()
        self._deliveries = {}

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
            # This installer is a worker-local dictionary update, with no
            # peer calls or waits. Uncertain installation remains recorded.
            self._install(request_id, lease, ledger)
            delivery.installed = True

    def delivery_status(self, lease: TargetLease) -> dict:
        with self._lock:
            delivery = self._deliveries.get(lease.request)
            if delivery is None or delivery.lease != lease or not delivery.installed:
                raise ContractError("Unknown or incompletely admitted target delivery")
            return {"ready": delivery.ledger.should_publish(), "remote_quiesced": delivery.ledger.remote_quiesced()}

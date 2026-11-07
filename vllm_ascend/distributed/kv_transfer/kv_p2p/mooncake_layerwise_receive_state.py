# SPDX-License-Identifier: Apache-2.0
"""Worker-local receive records, shared by message handling and owner admission.

The index owns its RLock. Delivery admission may acquire this lock while holding
its own lock; receive handlers must drop it before invoking provider/runtime
callbacks. Ledger operations may run under the index lock, never the reverse.
Native waits and RPCs do not run here. Ordinary P/D request maps remain separate.
"""

import threading
from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import ContractError
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_exact_receipt import ExactReceiptLedger


@dataclass(eq=False)
class ReceiveRecord:
    request_id: str
    external_id: str
    ledger: object
    deadline: float | None

    @property
    def instance(self):
        return self.ledger.request.request_instance

    @property
    def exact(self):
        return isinstance(self.ledger, ExactReceiptLedger)


class ReceiveRequestIndex:
    """One record per generation; secondary indexes contain identities only."""

    def __init__(self):
        self.lock = threading.RLock()
        self._records = {}
        self._by_request = {}
        self._mode_requests = set()

    def get(self, instance):
        if type(instance) is not str:
            return None
        with self.lock:
            return self._records.get(instance)

    def install(self, request_id, external_id, ledger, deadline):
        if type(request_id) is not str or not request_id or type(external_id) is not str or not external_id:
            raise ContractError("Receive records require explicit internal and external request IDs")
        record = ReceiveRecord(request_id, external_id, ledger, deadline)
        with self.lock:
            previous = self._records.get(record.instance)
            if previous is not None and (record.exact or previous.exact):
                raise ContractError("Target receive request identity is already installed")
            if record.exact and self._by_request.get(request_id):
                raise ContractError("Target internal request ID is already installed")
            if previous is not None:
                self._remove(previous)
            self._records[record.instance] = record
            self._by_request.setdefault(request_id, set()).add(record.instance)
            self._mode_requests.add(external_id)
        return record

    def retained_blocks(self, request_id):
        with self.lock:
            exact = [self._records[i] for i in self._by_request.get(request_id, ()) if self._records[i].exact]
            if not exact:
                return None
            if len(exact) != 1:
                raise ContractError("Ambiguous retained target request ID")
            return exact[0].ledger.lease.allocation.group_block_ids

    def in_receipt_mode(self, request_id):
        with self.lock:
            return request_id in self._mode_requests

    def clear_deadline(self, record):
        with self.lock:
            if self._records.get(record.instance) is record:
                record.deadline = None

    def expired(self, now):
        with self.lock:
            return tuple(r for r in self._records.values() if r.deadline is not None and now >= r.deadline)

    def _remove(self, record):
        del self._records[record.instance]
        instances = self._by_request[record.request_id]
        instances.remove(record.instance)
        if not instances:
            del self._by_request[record.request_id]

    def publish(self, record):
        with self.lock:
            if self._records.get(record.instance) is not record or record.exact and record.ledger.delivery_aborted:
                return False
            self._mode_requests.add(record.request_id)
            record.deadline = None
            if not record.exact:
                self._remove(record)  # Keep legacy mode classification until request end.
            return True

    def retire(self, instance, ledger):
        with self.lock:
            record = self._records.get(instance)
            if record is None or record.ledger is not ledger:
                return False  # A delayed retirement cannot erase another generation.
            self._remove(record)
            for alias in (record.request_id, record.external_id):
                if not any(alias in (r.request_id, r.external_id) for r in self._records.values()):
                    self._mode_requests.discard(alias)
            return True

    def forget(self, request_id, *, failed=False):
        with self.lock:
            for instance in tuple(self._by_request.get(request_id, ())):
                record = self._records[instance]
                if record.exact:
                    if failed:
                        record.ledger.abort_delivery()
                    continue  # Keep late DMA quiescence proof until owner retirement.
                self._remove(record)
            if request_id not in self._by_request:
                self._mode_requests.discard(request_id)

    def expire(self, record):
        with self.lock:
            if self._records.get(record.instance) is not record:
                return False
            if record.exact:
                record.ledger.abort_delivery()
                record.deadline = None
            else:
                self._remove(record)
            return True

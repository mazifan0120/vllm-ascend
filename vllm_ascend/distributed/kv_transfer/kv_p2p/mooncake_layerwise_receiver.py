# SPDX-License-Identifier: Apache-2.0
"""Receipt message handling, visibility and watchdog, independent of transport."""

import math
import time
from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    CONTRACT_VERSION,
    Component,
    ContractError,
    RequestKey,
    Status,
    TargetScope,
    TokenRange,
    TransferKey,
    TransferResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_exact_receipt import ExactReceiptLedger
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_protocol import (
    LAYER_RESULT_MSG,
    MAX_PROOF_BATCH_FRAMES,
    PROOF_ACK,
    PROOF_ACK_TARGET_CLOSED,
    RANK_TERMINAL_MSG,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_receive_state import ReceiveRequestIndex


class ReceiptReceiver:
    def __init__(self, timeout, logger, *, status=None):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("kv_connector_extra_config.layerwise_receipt_timeout_s must be positive and finite")
        self.timeout = float(timeout)
        self.logger = logger
        self.requests = ReceiveRequestIndex()
        self.lock = self.requests.lock
        self.status = (
            status
            if status is not None
            else SimpleNamespace(done_requests=set(), failed_requests=set(), receipt_provider=None)
        )

    def handle_proof_batch(self, msg: tuple) -> bytes:
        if (
            len(msg) != 3
            or type(msg[1]) is not int
            or msg[1] != CONTRACT_VERSION
            or type(msg[2]) not in (tuple, list)
            or not 0 < len(msg[2]) <= MAX_PROOF_BATCH_FRAMES
        ):
            raise ContractError("Malformed manifest completion proof batch")
        identity = None
        for frame in msg[2]:
            if (
                type(frame) not in (tuple, list)
                or len(frame) < 4
                or frame[0] not in (LAYER_RESULT_MSG, RANK_TERMINAL_MSG)
            ):
                raise ContractError("Manifest proof batch contains an unsupported operation")
            frame_identity = RequestKey(frame[2], frame[3])
            if identity is not None and frame_identity != identity:
                raise ContractError("Manifest proof batch mixes request generations")
            identity = frame_identity
        for frame in msg[2]:
            if frame[0] == LAYER_RESULT_MSG:
                self.handle_layer_result(tuple(frame))
            else:
                self.handle_rank_terminal(tuple(frame))
        with self.lock:
            record = self.requests.get(identity.request_instance)
            ledger = None if record is None else record.ledger
        if isinstance(ledger, ExactReceiptLedger) and ledger.request == identity and ledger.delivery_aborted:
            return PROOF_ACK_TARGET_CLOSED
        return PROOF_ACK

    def handle_layer_result(self, msg: tuple) -> None:
        """LAYER_RESULT_MSG: (type, version, instance, gen, rank,
        layer, op_id, component, start, end, backing_id, backing_gen, status,
        source_quiesced). Judged by identity in the ledger."""
        try:
            (
                _,
                version,
                instance,
                generation,
                rank,
                layer,
                op_id,
                component,
                start,
                end,
                backing_id,
                backing_gen,
                status,
                quiesced,
            ) = msg
        except Exception:
            self.logger.error("Malformed LAYER_RESULT_MSG (expected 14 fields): %r", msg)
            return
        if type(version) is not int or version != CONTRACT_VERSION:
            self.logger.error("LAYER_RESULT_MSG version mismatch: %s != %s", version, CONTRACT_VERSION)
            return
        with self.lock:
            record = self.requests.get(instance)
            ledger = None if record is None else record.ledger
        if ledger is None:
            self.logger.debug("LAYER_RESULT for unknown instance %s (plain-mode or stale); dropped", instance)
            return
        # P does not know D's backing layout in v1 (manifest transport is a
        # future rendezvous item): an empty backing_id resolves against the
        # ledger's own registered identities for the component. Multi-backing
        # layers need the manifest; unresolvable identities are dropped, never
        # fabricated.
        if not backing_id and not isinstance(ledger, ExactReceiptLedger):
            if component == Component.MAIN.value and ledger.main_backings:
                backing_id = ledger.main_backings[0]
            elif component == Component.INDEXER.value and ledger.indexer_backings:
                backing_id = ledger.indexer_backings[0]
        if not backing_id:
            self.logger.error(
                "LAYER_RESULT for %s carries an unresolvable empty backing (component=%s); dropped",
                instance,
                component,
            )
            return
        try:
            key = TransferKey(
                request=RequestKey(request_instance=instance, transfer_generation=generation),
                rank=rank,
                logical_layer=layer,
                operation_id=op_id,
                component=Component(component),
                token_range=TokenRange(start=start, end=end),
                target_scope=TargetScope(
                    backing_id=backing_id,
                    backing_generation=backing_gen,
                    consumer_rank=None if component == Component.MAIN.value else rank,
                ),
            )
            result = TransferResult(key=key, status=Status(status), source_quiesced=quiesced)
        except Exception as e:
            self.logger.error("Invalid LAYER_RESULT_MSG fields for %s: %s (msg=%r)", instance, e, msg)
            return
        ledger.record(result)
        self._maybe_publish(instance)

    def handle_rank_terminal(self, msg: tuple) -> None:
        """RANK_TERMINAL_MSG: (type, version, instance, gen, rank,
        status). Empty payloads must still terminal."""
        try:
            (_, version, instance, generation, rank, status) = msg[:6]
            if len(msg) not in (6, 7):
                raise ValueError("Unexpected terminal field count")
        except Exception:
            self.logger.error("Malformed RANK_TERMINAL_MSG (expected 6 fields): %r", msg)
            return
        if type(version) is not int or version != CONTRACT_VERSION:
            self.logger.error("RANK_TERMINAL_MSG version mismatch: %s != %s", version, CONTRACT_VERSION)
            return
        with self.lock:
            record = self.requests.get(instance)
            ledger = None if record is None else record.ledger
        if ledger is None:
            self.logger.warning("RANK_TERMINAL for unknown instance %s; dropped", instance)
            return
        try:
            request = RequestKey(instance, generation)
            if request != ledger.request:
                return
            if isinstance(ledger, ExactReceiptLedger):
                if len(msg) != 7:
                    raise ContractError("Manifest rank terminal is missing explicit DMA quiescence")
                ledger.finish_rank(request, rank, Status(status), msg[6])
            else:
                ledger.finish_rank(rank, Status(status))
        except (ContractError, ValueError) as error:
            self.logger.error("Invalid rank terminal for %s: %s", instance, error)
            return
        self._maybe_publish(instance)

    def _maybe_publish(self, instance: str) -> None:
        """Visibility flow, run in the receive thread (sole writer):
        data ready -> receipt(visible=False) -> provider.prepare_decode -> normal
        return -> mark visible -> done_requests. Any prepare_decode error means
        never visible, never published."""
        with self.lock:
            record = self.requests.get(instance)
            ledger = None if record is None else record.ledger
            req_id = None if record is None else record.request_id
            if ledger is None or req_id is None or req_id in self.status.done_requests:
                return
        receipt = ledger.build_receipt()
        if isinstance(ledger, ExactReceiptLedger):
            if ledger.delivery_aborted or ledger.should_publish():
                return
            if ledger.failed:
                ledger.abort_delivery()
                with self.lock:
                    self.status.failed_requests.add(req_id)
                    self.requests.clear_deadline(record)
                return
        if receipt is None:
            return  # data conditions not met yet
        provider = self.status.receipt_provider
        lease = getattr(ledger, "lease", None)
        if provider is None or lease is None:
            # A receipt-mode request can never fall back to plain-mode
            # publishing: without prepare_decode the consumer dependency is
            # never wired and the target bytes are not safe to decode. Fail
            # the request through the bounded-failure channel instead of
            # silently releasing it as done.
            self.logger.error(
                "Receipt-mode request %s (instance=%s) has no provider or lease; failing instead of publishing",
                req_id,
                instance,
            )
            with self.lock:
                self.status.failed_requests.add(req_id)
                self.requests.clear_deadline(record)
            return
        try:
            if isinstance(ledger, ExactReceiptLedger):
                provider.confirm_native_received(lease, receipt)
            provider.prepare_decode(lease, receipt)
        except Exception as e:
            self.logger.error(
                "prepare_decode failed for instance %s (req %s); not publishing. error=%s", instance, req_id, e
            )
            return
        with self.lock:
            # Provider operations run outside the index lock. Recheck that a
            # delayed callback cannot publish a retired or aborted generation.
            if (
                self.requests.get(instance) is not record
                or isinstance(ledger, ExactReceiptLedger)
                and ledger.delivery_aborted
            ):
                return
            ranks = (
                (ledger.visibility_rank,) if isinstance(ledger, ExactReceiptLedger) else sorted(ledger.expected_ranks)
            )
            try:
                for rank in ranks:
                    ledger.mark_visible(rank)
            except ContractError:
                if isinstance(ledger, ExactReceiptLedger) and ledger.delivery_aborted:
                    return  # Cancellation won the ledger lock; never publish.
                raise
            receipt.visible = True
            if not self.requests.publish(record):
                return
            self.status.done_requests.add(req_id)
        self.logger.info("Receipt-mode request %s (instance=%s) published after prepare_decode", req_id, instance)

    def forget_request(self, req_id, *, failed=False):
        self.requests.forget(req_id, failed=failed)

    def _sweep_receipt_deadlines(self):
        expired = []
        with self.lock:
            for record in self.requests.expired(time.monotonic()):
                if self.requests.expire(record):
                    self.status.failed_requests.add(record.request_id)
                    expired.append(record)
        for record in expired:
            self.logger.error(
                "Receipt watchdog: request %s (instance=%s) did not publish within %.0fs; failing bounded",
                record.request_id,
                record.instance,
                self.timeout,
            )

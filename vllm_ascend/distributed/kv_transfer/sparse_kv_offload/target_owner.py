# SPDX-License-Identifier: Apache-2.0
"""Worker-local owner facade and pure-data scheduler/worker dispatch boundary."""

import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from uuid import uuid4

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    CONTRACT_VERSION,
    VISIBILITY_CONTRACT,
    CacheView,
    ContractError,
    TargetLease,
    assert_lease_consistent,
    assert_provider_capabilities,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_catalog import (
    RegisteredTargetResources,
    TargetCatalog,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_runtime import (
    NpuTargetStreams,
    OwnerTargetRuntime,
    ReceivedTargetWrites,
)


class TargetOwner:
    contract_version = CONTRACT_VERSION
    visibility_contract = VISIBILITY_CONTRACT
    supports_layerwise_buffer_reuse = True

    def __init__(self, resources: RegisteredTargetResources, runtime: OwnerTargetRuntime) -> None:
        self.resources = resources
        self.runtime = runtime
        self.provider = resources.build_provider(runtime)

    def catalog(self) -> TargetCatalog:
        return self.resources.catalog()

    def acquire_target(self, request, allocation) -> TargetLease:
        return self.provider.acquire_target(request, allocation)

    def get_target_views(self, lease) -> tuple[CacheView, ...]:
        return self.provider.get_target_views(lease)

    def prepare_decode(self, lease, receipt) -> None:
        self.provider.prepare_decode(lease, receipt)

    def release_target(self, lease) -> None:
        self.provider.release_target(lease)

    def retry_failed_retains(self) -> None:
        self.runtime.retry_failed_retains()

    def resolve_target_regions(self, lease):
        # Data-plane-only address handles are deliberately not an owner IPC
        # operation. The scheduler continues to receive identities/offsets.
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import RegionAddress

        with self.provider.inspect_target(lease):
            self.resources.validate(lease)
            return tuple(
                RegionAddress(
                    lease.request,
                    lease.allocation.allocation_generation,
                    lease.allocation.retention_token,
                    region,
                    self.resources._base_addresses[region.region_id],
                )
                for region in lease.regions
            )

    def publish_received(self, lease, host_visible, indexer_event) -> None:
        self.runtime.publish_received(lease, ReceivedTargetWrites(lease, host_visible, indexer_event))

    @contextmanager
    def use(self, lease):
        with self.runtime.use(lease):
            yield

    def close(self) -> None:
        self.runtime.assert_can_close()
        self.resources.close()

    def __getstate__(self):
        raise TypeError("Target owner contains local allocations and must not be serialized")


class DeferredTargetProvider:
    """Stable injection identity before KV allocation/TE registration finishes."""

    contract_version = CONTRACT_VERSION
    visibility_contract = VISIBILITY_CONTRACT
    supports_layerwise_buffer_reuse = True

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owner: TargetOwner | None = None
        self._operations = 0
        self._closing = False
        self._closed = False

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._owner is not None and not self._closed and not self._closing

    def bind(self, owner: TargetOwner) -> None:
        assert_provider_capabilities(owner)
        with self._lock:
            if self._closed or self._closing or self._owner is not None or self._operations:
                raise ContractError("Target facade cannot be rebound across registration generations")
            self._owner = owner

    @contextmanager
    def _operation(self):
        with self._lock:
            if self._owner is None or self._closed or self._closing:
                raise ContractError("Target owner catalog is not registered and ready")
            self._operations += 1
            owner = self._owner
        try:
            yield owner
        finally:
            with self._lock:
                self._operations -= 1

    def catalog(self) -> TargetCatalog:
        with self._operation() as owner:
            return owner.catalog()

    def acquire_target(self, request, allocation) -> TargetLease:
        with self._operation() as owner:
            return owner.acquire_target(request, allocation)

    def get_target_views(self, lease) -> tuple[CacheView, ...]:
        with self._operation() as owner:
            return owner.get_target_views(lease)

    def resolve_target_regions(self, lease):
        with self._operation() as owner:
            return owner.resolve_target_regions(lease)

    def prepare_decode(self, lease, receipt) -> None:
        with self._operation() as owner:
            owner.prepare_decode(lease, receipt)

    def release_target(self, lease) -> None:
        with self._operation() as owner:
            owner.release_target(lease)

    def retry_failed_retains(self) -> None:
        """Owner-local cleanup; not a remote wire operation or DMA ack."""
        with self._operation() as owner:
            owner.retry_failed_retains()

    def publish_received(self, lease, host_visible, indexer_event) -> None:
        with self._operation() as owner:
            owner.publish_received(lease, host_visible, indexer_event)

    @contextmanager
    def use(self, lease):
        with self._operation() as owner, owner.use(lease):
            yield

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._closing or self._operations:
                raise ContractError("Target facade still has active owner operations")
            self._closing = True
            owner = self._owner
        try:
            if owner is not None:
                owner.close()
        except Exception:
            with self._lock:
                self._closing = False
            raise
        with self._lock:
            self._closed = True
            self._closing = False

    def __getstate__(self):
        raise TypeError("Use TargetProviderClient across processes, not a pickled local facade")


def create_registered_target_owner(
    manager,
    kv_caches,
    te,
    *,
    components,
    layer_groups,
    block_tokens,
    num_blocks,
    retention,
    device_id,
    consumption_stream,
    timeout,
):
    """A2 binding after actual manager allocation and successful TE registration."""
    resources = RegisteredTargetResources(
        manager,
        kv_caches,
        te,
        components=components,
        layer_groups=layer_groups,
        block_tokens=block_tokens,
        num_blocks=num_blocks,
    )
    try:
        streams = NpuTargetStreams(manager, device_id, consumption_stream)
        runtime = OwnerTargetRuntime(resources, retention, streams, timeout=timeout)
        return TargetOwner(resources, runtime)
    except Exception:
        resources.close()
        raise


@dataclass(frozen=True)
class TargetOwnerCall:
    call_id: str
    operation: str
    arguments: tuple


@dataclass(frozen=True)
class TargetOwnerReply:
    call_id: str
    result: object = None
    error_type: str = ""
    error_message: str = ""


def _assert_wire_value(value) -> None:
    if value is None or isinstance(value, (str, int, bool, Enum)):
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_wire_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_wire_value(key)
            _assert_wire_value(item)
        return
    if is_dataclass(value) and value.__class__.__module__ in (
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract",
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_provider",
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_catalog",
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_manifest",
        __name__,
    ):
        for field in fields(value):
            _assert_wire_value(getattr(value, field.name))
        return
    raise TypeError("Owner IPC accepts only typed catalog/lease/receipt values, not local runtime objects")


class TargetOwnerDispatcher:
    """Run in the allocation owner, behind B's existing serialized IPC channel.

    Neither receive fences nor use contexts cross this boundary. The real D
    worker installs them locally and the scheduler only receives pure values.
    """

    def __init__(self, facade: DeferredTargetProvider, *, delivery=None) -> None:
        self._facade = facade
        self._delivery = delivery

    def dispatch(self, call: TargetOwnerCall) -> TargetOwnerReply:
        if not isinstance(call, TargetOwnerCall) or not call.call_id:
            raise ContractError("Malformed target owner command")
        operations = {
            "catalog": self._facade.catalog,
            "acquire": self._facade.acquire_target,
            "views": self._facade.get_target_views,
            "prepare": self._facade.prepare_decode,
            "release": self._facade.release_target,
        }
        for operation in ("manifest", "catalogs"):
            method = getattr(self._facade, operation, None)
            if callable(method):
                operations[operation] = method
        for operation in ("admit_target", "delivery_status"):
            method = getattr(self._delivery if self._delivery is not None else self._facade, operation, None)
            if callable(method):
                operations[operation] = method
        try:
            _assert_wire_value(call.arguments)
            if call.operation not in operations:
                raise ContractError("Unsupported target owner operation")
            result = deepcopy(operations[call.operation](*call.arguments))
            _assert_wire_value(result)
            return TargetOwnerReply(call.call_id, result)
        except Exception as error:
            return TargetOwnerReply(call.call_id, error_type=type(error).__name__, error_message=str(error))


class TargetProviderClient:
    """Four-method facade over a real scheduler-worker transport.call().

    The transport must dispatch on the owner and bound its waits. No transport
    is guessed or silently replaced by an in-process default.
    """

    contract_version = CONTRACT_VERSION
    visibility_contract = VISIBILITY_CONTRACT
    supports_layerwise_buffer_reuse = True

    def __init__(self, transport=None) -> None:
        self._transport = None
        self._lock = threading.Lock()
        if transport is not None:
            self.bind_transport(transport)

    def bind_transport(self, transport) -> None:
        if not callable(getattr(transport, "call", None)):
            raise TypeError("TargetProviderClient requires a real owner transport")
        with self._lock:
            if self._transport is not None:
                raise ContractError("Target owner transport cannot be rebound across owner generations")
            self._transport = transport

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._transport is not None

    def _call(self, operation, *arguments):
        _assert_wire_value(arguments)
        call = TargetOwnerCall(uuid4().hex, operation, arguments)
        with self._lock:
            transport = self._transport
        if transport is None:
            raise ContractError("Target owner transport is not connected")
        reply = transport.call(call)
        if not isinstance(reply, TargetOwnerReply) or reply.call_id != call.call_id:
            raise ContractError("Target owner response belongs to another IPC operation")
        if reply.error_type:
            error = ContractError if reply.error_type == "ContractError" else RuntimeError
            raise error(f"Target owner {reply.error_type}: {reply.error_message}")
        _assert_wire_value(reply.result)
        return reply.result

    def catalog(self) -> TargetCatalog:
        result = self._call("catalog")
        if not isinstance(result, TargetCatalog) or result.contract_version != CONTRACT_VERSION:
            raise ContractError("Target owner catalog protocol mismatch")
        return result

    def acquire_target(self, request, allocation) -> TargetLease:
        result = self._call("acquire", request, allocation)
        if not isinstance(result, TargetLease):
            raise ContractError("Target owner did not return a lease")
        assert_lease_consistent(result, request)
        if result.allocation != allocation:
            raise ContractError("Target owner returned another scheduler allocation")
        return result

    def get_target_views(self, lease) -> tuple[CacheView, ...]:
        result = self._call("views", lease)
        if not isinstance(result, tuple) or result != lease.views or any(not isinstance(v, CacheView) for v in result):
            raise ContractError("Target owner did not return flat cache views")
        return result

    def get_target_manifest(self, lease):
        # Lazy import avoids the owner/catalog/manifest initialization cycle.
        from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_manifest import TargetManifest

        result = self._call("manifest", lease)
        if type(result) is not TargetManifest or result.lease != lease:
            raise ContractError("Target owner did not return the matching rank-explicit manifest")
        return result

    def admit_target(self, manifest, request_id) -> None:
        if self._call("admit_target", manifest, request_id) is not None:
            raise ContractError("Target owner admission returned a malformed acknowledgement")

    def delivery_status(self, lease) -> dict:
        result = self._call("delivery_status", lease)
        if (
            type(result) is not dict
            or set(result) != {"ready", "remote_quiesced"}
            or any(type(value) is not bool for value in result.values())
        ):
            raise ContractError("Target owner returned malformed delivery proofs")
        return result

    def prepare_decode(self, lease, receipt) -> None:
        if self._call("prepare", lease, receipt) is not None:
            raise ContractError("Target owner prepare_decode returned a malformed acknowledgement")

    def release_target(self, lease) -> None:
        if self._call("release", lease) is not None:
            raise ContractError("Target owner release_target returned a malformed acknowledgement")

    def __getstate__(self):
        raise TypeError("TargetProviderClient transport stays in its calling process")

# SPDX-License-Identifier: Apache-2.0
"""Explicit JSON data for the D-to-P target lease publication.

An HTTP metaserver cannot carry WriterPlan/TargetLease Python instances.
Keep the network schema separate from owner-local resources and reconstruct
the contract types before a producer consumes the descriptor.
"""

from dataclasses import fields

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    CONTRACT_VERSION,
    BlockAllocation,
    CacheView,
    Component,
    ContractError,
    RegisteredRegion,
    RequestKey,
    TargetLease,
    TokenRange,
    WriterPlan,
    assert_lease_consistent,
)

TARGET_LEASE_WIRE_VERSION = 1


def _plain(value):
    """Copy JSON values without calling arbitrary serializers or tensor APIs."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) in (list, tuple):
        return [_plain(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _plain(item) for key, item in value.items()}
    raise ContractError(f"Target wire metadata contains a non-JSON value: {type(value).__name__}")


def _object(value, names, description):
    if type(value) is not dict or set(value) != set(names):
        raise ContractError(f"Malformed target wire {description}: expected fields {tuple(names)}")
    return value


def _items(value, description):
    if type(value) is not list:
        raise ContractError(f"Target wire {description} must be a list")
    return value


def _record(value):
    return {field.name: getattr(value, field.name) for field in fields(value)}


def target_lease_to_wire(lease: TargetLease) -> dict:
    """Publish identities, geometry and block IDs; never owner-local pointers."""
    if not isinstance(lease, TargetLease):
        raise ContractError("Target publication requires a TargetLease")
    if not lease.allocation.retention_token:
        raise ContractError("Target publication requires scheduler retention")
    assert_lease_consistent(lease, lease.request)
    views = []
    for view in lease.views:
        record = _record(view)
        record["component"] = view.component.value
        views.append(record)
    payload = {
        "wire_version": TARGET_LEASE_WIRE_VERSION,
        "contract_version": lease.contract_version,
        "request": _record(lease.request),
        "rank": lease.rank,
        "allocation": {**_record(lease.allocation), "request": _record(lease.allocation.request)},
        "regions": [_record(region) for region in lease.regions],
        "views": views,
        "writer_plan": {
            "writers_by_backing": lease.writer_plan.writers_by_backing,
            "ranges_by_backing": {
                backing: [_record(span) for span in spans]
                for backing, spans in lease.writer_plan.ranges_by_backing.items()
            },
        },
    }
    result = _plain(payload)
    # Apply the same validation in both directions, including constructor
    # checks that a WriterPlan's inner dictionaries do not perform themselves.
    if target_lease_from_wire(result) != lease:
        raise ContractError("Target publication changed the acquired lease")
    return result


def target_lease_from_wire(payload: dict) -> TargetLease:
    """Validate a metaserver JSON descriptor before using a remote target."""
    value = _object(
        _plain(payload),
        ("wire_version", "contract_version", "request", "rank", "allocation", "regions", "views", "writer_plan"),
        "lease",
    )
    if type(value["wire_version"]) is not int or value["wire_version"] != TARGET_LEASE_WIRE_VERSION:
        raise ContractError("Unsupported target lease wire version")
    if type(value["contract_version"]) is not int or value["contract_version"] != CONTRACT_VERSION:
        raise ContractError("Unsupported target lease contract version")
    request_fields = tuple(field.name for field in fields(RequestKey))
    request = RequestKey(**_object(value["request"], request_fields, "request"))
    allocation = dict(
        _object(value["allocation"], tuple(field.name for field in fields(BlockAllocation)), "allocation")
    )
    allocation["request"] = RequestKey(**_object(allocation["request"], request_fields, "allocation request"))
    allocation = BlockAllocation(**allocation)
    if not allocation.retention_token:
        raise ContractError("Published target allocation has no retention token")
    regions = tuple(
        RegisteredRegion(**_object(region, tuple(field.name for field in fields(RegisteredRegion)), "region"))
        for region in _items(value["regions"], "regions")
    )
    views = []
    for record in _items(value["views"], "views"):
        record = dict(_object(record, tuple(field.name for field in fields(CacheView)), "view"))
        try:
            record["component"] = Component(record["component"])
        except (TypeError, ValueError) as error:
            raise ContractError("Unknown target view component") from error
        view = CacheView(**record)
        if not view.strides_bytes or any(type(stride) is not int or stride < 1 for stride in view.strides_bytes):
            raise ContractError("Target view strides must be positive integer byte strides")
        if any(type(block) is not int or block < 0 or block >= view.block_capacity for block in view.block_ids):
            raise ContractError("Target view contains an invalid block ID")
        views.append(view)
    plan = _object(value["writer_plan"], ("writers_by_backing", "ranges_by_backing"), "writer plan")
    if type(plan["writers_by_backing"]) is not dict or type(plan["ranges_by_backing"]) is not dict:
        raise ContractError("Target writer plan must contain backing maps")
    writers = {}
    for backing, ranks in plan["writers_by_backing"].items():
        ranks = _items(ranks, "writer ranks")
        if not backing or not ranks or any(type(rank) is not int or rank < 0 for rank in ranks):
            raise ContractError("Target writer plan contains invalid ranks")
        if len(set(ranks)) != len(ranks):
            raise ContractError("Target writer plan repeats a writer rank")
        writers[backing] = tuple(ranks)
    ranges = {
        backing: tuple(
            TokenRange(**_object(span, ("start", "end"), "writer range")) for span in _items(spans, "ranges")
        )
        for backing, spans in plan["ranges_by_backing"].items()
    }
    if set(writers) != set(ranges):
        raise ContractError("Target writer ranks and ranges describe different backings")
    lease = TargetLease(
        value["contract_version"],
        request,
        value["rank"],
        allocation,
        regions,
        tuple(views),
        WriterPlan(writers, ranges),
    )
    assert_lease_consistent(lease, request)
    return lease

# SPDX-License-Identifier: Apache-2.0
"""Explicit value-only encoding for independent scheduler/worker processes.

This protocol never imports a type named by a peer. Its fixed whitelist omits
LocalRegion, tensors, streams, guards and owners. Dict keys retain their enum
identity; tuples retain their shape. Constructor validation still runs after
decoding. Transport limits apply before JSON parsing and during tree traversal.
"""

import json
from dataclasses import fields

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    BlockAllocation,
    CacheView,
    Component,
    ContractError,
    ReceiveReceipt,
    RegisteredRegion,
    ReleaseAck,
    RequestKey,
    Status,
    TargetLease,
    TargetScope,
    TokenRange,
    TransferKey,
    TransferResult,
    WriterPlan,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_catalog import TargetCatalog, TargetEntry
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_manifest import RankTarget, TargetManifest
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import TargetOwnerCall, TargetOwnerReply
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_wire import RetentionOp, RetentionResult
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_provider import TargetBinding

OWNER_WIRE_VERSION = 1
MAX_OWNER_FRAME_BYTES = 8 * 1024 * 1024
MAX_OWNER_WIRE_DEPTH = 64
MAX_OWNER_WIRE_NODES = 500_000

# Immutable whitelists: neither mutable process rendezvous nor dynamic imports.
OWNER_RECORD_TYPES = (
    BlockAllocation,
    CacheView,
    ReceiveReceipt,
    RegisteredRegion,
    ReleaseAck,
    RequestKey,
    TargetLease,
    TargetScope,
    TokenRange,
    TransferKey,
    TransferResult,
    WriterPlan,
    TargetBinding,
    TargetCatalog,
    TargetEntry,
    TargetOwnerCall,
    TargetOwnerReply,
    RetentionOp,
    RetentionResult,
    RankTarget,
    TargetManifest,
)
OWNER_ENUM_TYPES = (Component, Status)


class _ValueCodec:
    def __init__(self) -> None:
        self._remaining = MAX_OWNER_WIRE_NODES

    def _visit(self, depth: int) -> None:
        self._remaining -= 1
        if depth > MAX_OWNER_WIRE_DEPTH or self._remaining < 0:
            raise ContractError("Owner wire value exceeds traversal limits")

    def encode(self, value, depth=0):
        self._visit(depth)
        kind = type(value)
        if value is None or kind in (str, int, bool):
            return value
        if kind in OWNER_ENUM_TYPES:
            return ["enum", kind.__name__, value.value]
        if kind in (tuple, list):
            return ["tuple" if kind is tuple else "list", [self.encode(v, depth + 1) for v in value]]
        if kind is dict:
            return ["dict", [[self.encode(k, depth + 1), self.encode(v, depth + 1)] for k, v in value.items()]]
        if kind in OWNER_RECORD_TYPES:
            return ["record", kind.__name__, [self.encode(getattr(value, f.name), depth + 1) for f in fields(kind)]]
        raise ContractError("Owner wire accepts only explicitly whitelisted protocol values")

    def decode(self, value, depth=0):
        self._visit(depth)
        if value is None or type(value) in (str, int, bool):
            return value
        if type(value) is not list or not value or type(value[0]) is not str:
            raise ContractError("Malformed owner wire value")
        tag = value[0]
        if tag == "enum" and len(value) == 3:
            kind = next((t for t in OWNER_ENUM_TYPES if t.__name__ == value[1]), None)
            if kind is None or type(value[2]) is not str:
                raise ContractError("Unknown owner wire enum")
            return kind(value[2])
        if tag == "record" and len(value) == 3:
            kind = next((t for t in OWNER_RECORD_TYPES if t.__name__ == value[1]), None)
            if kind is None or type(value[2]) is not list or len(value[2]) != len(fields(kind)):
                raise ContractError("Unknown or malformed owner wire record")
            return kind(*(self.decode(v, depth + 1) for v in value[2]))
        if tag in ("tuple", "list", "dict") and len(value) == 2 and type(value[1]) is list:
            if tag == "dict":
                result = {}
                for pair in value[1]:
                    if type(pair) is not list or len(pair) != 2:
                        raise ContractError("Malformed owner wire dictionary entry")
                    key, item = (self.decode(v, depth + 1) for v in pair)
                    if key in result:
                        raise ContractError("Duplicate owner wire dictionary key")
                    result[key] = item
                return result
            result = [self.decode(v, depth + 1) for v in value[1]]
            return tuple(result) if tag == "tuple" else result
        raise ContractError("Unknown owner wire value tag")


def encode_owner_value(value) -> bytes:
    try:
        frame = json.dumps([OWNER_WIRE_VERSION, _ValueCodec().encode(value)], separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ContractError(f"Cannot encode owner wire value: {error}") from error
    if len(frame) > MAX_OWNER_FRAME_BYTES:
        raise ContractError("Owner wire frame exceeds size limit")
    return frame


def decode_owner_value(frame: bytes):
    if type(frame) is not bytes or len(frame) > MAX_OWNER_FRAME_BYTES:
        raise ContractError("Invalid owner wire frame size or type")
    try:
        envelope = json.loads(frame)
        if (
            type(envelope) is not list
            or len(envelope) != 2
            or type(envelope[0]) is not int
            or envelope[0] != OWNER_WIRE_VERSION
        ):
            raise ContractError("Owner wire protocol version mismatch")
        return _ValueCodec().decode(envelope[1])
    except (TypeError, ValueError, RecursionError, UnicodeError, KeyError, AttributeError, IndexError) as error:
        raise ContractError(f"Cannot decode owner wire value: {error}") from error

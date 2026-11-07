# SPDX-License-Identifier: Apache-2.0
"""Explicit value-only encoding for independent scheduler/worker processes.

This protocol never imports a type named by a peer. Its fixed whitelist omits
LocalRegion, tensors, streams, guards and owners. Dict keys retain their enum
identity; tuples retain their shape. Constructor validation still runs after
decoding. Transport limits apply before JSON parsing and during tree traversal.
"""

import json
from dataclasses import fields
from types import MappingProxyType

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
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_provider import TargetBinding
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention_protocol import RetentionOp, RetentionResult

OWNER_WIRE_VERSION = 2
SUPPORTED_OWNER_WIRE_VERSIONS = (1, OWNER_WIRE_VERSION)
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
_OWNER_RECORD_FIELDS = MappingProxyType(
    {kind: tuple(field.name for field in fields(kind)) for kind in OWNER_RECORD_TYPES}
)
_OWNER_RECORDS_BY_NAME = MappingProxyType({kind.__name__: kind for kind in OWNER_RECORD_TYPES})
_OWNER_ENUMS_BY_NAME = MappingProxyType({kind.__name__: kind for kind in OWNER_ENUM_TYPES})


class _ValueCodec:
    def __init__(self, *, references=True) -> None:
        self._remaining = MAX_OWNER_WIRE_NODES
        self._references = references
        self._encoded_tuples = {}
        self._encoded_tuple_ids = {}
        self._decoded_tuples = {}

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
        if self._references and kind is tuple and id(value) in self._encoded_tuple_ids:
            if value and depth >= MAX_OWNER_WIRE_DEPTH:
                raise ContractError("Owner wire value exceeds traversal limits")
            return ["tuple_ref", self._encoded_tuple_ids[id(value)][1]]
        if self._references and kind is tuple and all(type(v) in (str, int, bool, type(None)) for v in value):
            # Block IDs recur in every view and rank. Only immutable primitive
            # tuples can be referenced; lists/records/local objects never can.
            # Include each primitive's type: Python equates (True,) and (1,).
            identity = tuple((type(v), v) for v in value)
            if identity in self._encoded_tuples:
                if value and depth >= MAX_OWNER_WIRE_DEPTH:
                    raise ContractError("Owner wire value exceeds traversal limits")
                index = self._encoded_tuples[identity]
                self._encoded_tuple_ids[id(value)] = (value, index)
                return ["tuple_ref", index]
            index = len(self._encoded_tuples)
            self._encoded_tuples[identity] = index
            self._encoded_tuple_ids[id(value)] = (value, index)
            return ["tuple_def", index, [self.encode(v, depth + 1) for v in value]]
        if kind in (tuple, list):
            return ["tuple" if kind is tuple else "list", [self.encode(v, depth + 1) for v in value]]
        if kind is dict:
            return ["dict", [[self.encode(k, depth + 1), self.encode(v, depth + 1)] for k, v in value.items()]]
        record_fields = _OWNER_RECORD_FIELDS.get(kind)
        if record_fields is not None:
            return ["record", kind.__name__, [self.encode(getattr(value, name), depth + 1) for name in record_fields]]
        raise ContractError("Owner wire accepts only explicitly whitelisted protocol values")

    def decode(self, value, depth=0):
        self._visit(depth)
        if value is None or type(value) in (str, int, bool):
            return value
        if type(value) is not list or not value or type(value[0]) is not str:
            raise ContractError("Malformed owner wire value")
        tag = value[0]
        if self._references and tag in ("tuple_ref", "tuple_def"):
            if len(value) < 2 or type(value[1]) is not int or value[1] < 0:
                raise ContractError("Malformed owner primitive tuple reference")
            index = value[1]
            if tag == "tuple_ref":
                if len(value) != 2 or index not in self._decoded_tuples:
                    raise ContractError("Unknown or forward owner primitive tuple reference")
                if self._decoded_tuples[index] and depth >= MAX_OWNER_WIRE_DEPTH:
                    raise ContractError("Owner wire value exceeds traversal limits")
                return self._decoded_tuples[index]
            if (
                len(value) != 3
                or index != len(self._decoded_tuples)
                or type(value[2]) is not list
                or any(type(item) not in (str, int, bool, type(None)) for item in value[2])
            ):
                raise ContractError("Duplicate, unordered or non-primitive owner tuple definition")
            result = tuple(self.decode(item, depth + 1) for item in value[2])
            self._decoded_tuples[index] = result
            return result
        if tag == "enum" and len(value) == 3:
            kind = _OWNER_ENUMS_BY_NAME.get(value[1]) if type(value[1]) is str else None
            if kind is None or type(value[2]) is not str:
                raise ContractError("Unknown owner wire enum")
            return kind(value[2])
        if tag == "record" and len(value) == 3:
            kind = _OWNER_RECORDS_BY_NAME.get(value[1]) if type(value[1]) is str else None
            if kind is None or type(value[2]) is not list or len(value[2]) != len(_OWNER_RECORD_FIELDS[kind]):
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
            or envelope[0] not in SUPPORTED_OWNER_WIRE_VERSIONS
        ):
            raise ContractError("Owner wire protocol version mismatch")
        return _ValueCodec(references=envelope[0] >= 2).decode(envelope[1])
    except (TypeError, ValueError, RecursionError, UnicodeError, KeyError, AttributeError, IndexError) as error:
        raise ContractError(f"Cannot decode owner wire value: {error}") from error

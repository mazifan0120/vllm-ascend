# SPDX-License-Identifier: Apache-2.0
"""Plain layerwise connector metadata, importable without the NPU stack.

Keep optional transport extensions after legacy fields so old positional
constructors remain valid. Runtime owners, locks and completion events do not
belong in this request metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LayerMetadata:
    tensor_group_idx: list[int]
    kv_caches_base_addr: list[int]
    block_len: list[int]
    block_size_scale: list[int]


@dataclass
class ReqMeta:
    local_block_ids: list[list[int]]
    token_ids: list[int] | None
    remote_block_ids: list[list[int]]
    remote_block_size: list[list[int]]
    remote_engine_id: str | None
    remote_host: str | None
    remote_port: int | None
    remote_te_rpc_port: int | None
    remote_layer_metadata: dict[str, LayerMetadata] | None
    metaserver: str | None
    remote_tp_size: int | None
    remote_pcp_size: int | None
    remote_dcp_size: int | None
    chunk_finish: bool = False
    prompt_len: int = 0
    trans_count: list[int] | None = None
    remote_cache_tokens: int = 0
    local_computed_tokens: int = 0
    local_transed_tokens: int = 0
    do_virtual: bool = False
    request_instance: str = ""
    remote_te_rpc_endpoint: str | None = None
    remote_writer_plan: Any = None
    restore_only: bool = False
    restore_end_token: int = 0
    token_len_chunk: int = 0
    remote_target_lease: Any = None
    remote_target_manifest: Any = None

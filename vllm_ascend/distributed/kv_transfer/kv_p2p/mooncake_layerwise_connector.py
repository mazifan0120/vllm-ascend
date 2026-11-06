# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import hashlib
import logging
import math
import os
import queue
import struct
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import numpy as np
import numpy.typing as npt
import torch
import torch_npu
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
    get_world_group,
)
from vllm.logger import logger
from vllm.utils.math_utils import round_down
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import extract_layer_index

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import GET_META_MSG
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_transfer import (
    DsaRegisterAtom,
    collect_bounded_register_regions,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    CONTRACT_VERSION,
    BlockAllocation,
    Component,
    ContractError,
    ReleaseAck,
    RequestKey,
    Status,
    TargetLease,
    TargetScope,
    TokenRange,
    TransferKey,
    TransferResult,
    assert_lease_consistent,
    assert_pd_consistent,
    assert_provider_capabilities,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_exact_receipt import ExactReceiptLedger
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_metadata import LayerMetadata, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_peer import (
    GET_TARGET_REGIONS_MSG,
    LAYER_RESULTS_MSG,
    MAX_PROOF_BATCH_FRAMES,
    PROOF_ACK,
    PROOF_ACK_TARGET_CLOSED,
    TARGET_PEER_ERROR,
    ManifestPeerProofs,
    encode_target_addresses,
    resolve_manifest_addresses,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_receipt import (
    ReceiptLedger,
    SlotError,
    SlotLedger,
    is_no_compute_layer,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_wire import (
    target_lease_from_wire,
    target_lease_to_wire,
    target_manifest_from_wire,
    target_manifest_to_wire,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_writer import ManifestSourceWriter, SourceWriteJob
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_manifest import TargetManifest
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    PD_QOS_DEFAULT,
    RegisterRegions,
    align_memory,
    collect_storage_merged_register_regions,
    context_parallel_parameters_check,
    get_cp_group,
    get_local_remote_block_port_mappings,
    get_transfer_mappings,
    get_transfer_timeout_value,
    inject_qos,
    kv_alltoall_and_rearrange,
    parallel_info,
    validate_register_region_count,
)
from vllm_ascend.distributed.utils import get_decode_context_model_parallel_rank
from vllm_ascend.utils import (
    get_kv_cache_tensor_layers,
    npu_stream_switch,
    trans_nd_to_nz,
)

# isort: off
if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

DONE_SENDING_MSG = b"done_sending_msg"
FAILED_SENDING_MSG = b"failed_sending_msg"


# Layerwise receipt protocol positional arrays decoded by
# msgspec.Decoder(type=tuple), contract version at index 1. P emits one
# LAYER_RESULT per (layer, component, chunk span) plus one RANK_TERMINAL per
# request after the last layer (empty payloads included); D judges readiness
# by identity instead of counting DONE messages.
LAYER_RESULT_MSG = b"layer_result_msg"
RANK_TERMINAL_MSG = b"rank_terminal_msg"
LAYERWISE_REUSE_WAIT_LOG_INTERVAL_SECONDS = 10.0
# Upper bound for the slot-ledger reuse wait. The legacy event path
# waits unbounded with a log heartbeat; the ledger path needs an explicit
# deadline so a stalled transfer surfaces as an error instead of hanging.
LAYERWISE_SLOT_WAIT_TIMEOUT_SECONDS = 300.0
# Receipt watchdog (B5): a receipt-mode request whose ledger was installed
# but never published within this deadline fails loudly through the
# bounded-failure channel instead of hanging forever.
RECEIPT_WATCHDOG_SECONDS = 300.0


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    te_rpc_port: int
    layer_metadata: dict[str, LayerMetadata]


@dataclass
class SendTask:
    send_request: dict[str, ReqMeta] = field(default_factory=dict)
    # pd_head_ratio == 1 use
    wait_event: torch.npu.Event | None = None
    # pd_head_ratio > 1 use
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None
    # kv cache quantization layer use
    k_quant_cache: torch.Tensor | None = None
    v_quant_cache: torch.Tensor | None = None
    layer_idx: int = 0
    layer_name: str = ""
    layer_names: list[str] = field(default_factory=list)
    failed_requests: dict[str, ReqMeta] = field(default_factory=dict)
    # trans block info
    group_rearrange_block_ids: list[list[int]] | None = None
    group_num_blocks: list[int] | None = None
    group_num_tokens: list[int] | None = None
    group_block_table: list[torch.Tensor | None] | None = None
    group_block_len_tensor: list[torch.Tensor | None] | None = None
    group_seq_start_tensor: list[torch.Tensor | None] | None = None
    # Additional Store source fence; preserve wait_event for the legacy
    # attention/quantization/resharding path instead of replacing it.
    source_ready_event: Any = None
    source_completion: Any = None


@dataclass
class TransferMeta:
    src: list[int]
    dst: list[int]
    length: list[int]
    req_ids: list[str]


@dataclass
class SendReqInfo:
    local_block_ids: list[list[int]]
    local_transferred_tokens: int
    local_computed_tokens: int
    request: "Request"
    remote_target: tuple[TargetLease | None, TargetManifest | None] | None = None

    def extend_local_block_ids(self, new_block_ids: list[list[int]]) -> None:
        """extend local block ids for this step"""
        for i, new_block_id in enumerate(new_block_ids):
            self.local_block_ids[i].extend(new_block_id)

    def update_computed_tokens(self, computed_tokens: int) -> None:
        """update local computen tokens for this step"""
        self.local_computed_tokens = computed_tokens

    def update_transferred_tokens(self, transferred_tokens: int) -> None:
        """update transferred tokens for this step"""
        self.local_transferred_tokens = transferred_tokens

    def unpack(self):
        return (
            self.local_block_ids,
            self.local_transferred_tokens,
            self.local_computed_tokens,
            self.request,
        )


@dataclass
class SizedDict(OrderedDict):
    def __init__(self, max_size=16000, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            value: dict[int, list[int]] = {}
            self[key] = value
            return value


class KVCacheSendingLayerThread(threading.Thread):
    def __init__(
        self,
        engine: TransferEngine,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        kv_cache_specs: list[KVCacheSpec],
        attn_resharding_group_idx: set,
        total_layers: int,
        ready_event: threading.Event,
        tp_size: int,
        tp_rank: int,
        pd_head_ratio: int,
        num_head_replica: int,
        layer_metadata: dict[str, LayerMetadata],
        use_mla: bool,
        use_attn_mamba_hybrid: bool,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        enable_kv_quant: bool,
        enable_c8_quant: bool,
        resharding_stream: torch.npu.Stream,
        callback_func: Callable[..., None] = lambda x: None,
        reuse_completion_callback: Callable[[int, str | None], None] | None = None,
    ):
        super().__init__(daemon=True, name="KVCacheSendingLayerThread")
        self.engine = engine
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.kv_cache_specs = kv_cache_specs
        self.attn_resharding_group_idx = attn_resharding_group_idx
        self.mamba_cache_mode = self.vllm_config.cache_config.mamba_cache_mode
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pd_head_ratio = pd_head_ratio
        self.num_head_replica = num_head_replica
        self.layer_metadata = layer_metadata
        self.total_layers = total_layers
        self.use_mla = use_mla
        self.use_attn_mamba_hybrid = use_attn_mamba_hybrid
        self.resharding_stream = resharding_stream
        self.current_layer = -1

        send_queue_size = 0
        if self.pd_head_ratio != 1:
            if self.use_attn_mamba_hybrid:
                send_queue_size = len(self.kv_cache_specs)
            else:
                send_queue_size = 1
        self.send_queue = queue.Queue[SendTask](maxsize=send_queue_size)
        self.failed_reqs: set[str] = set()
        # Requests whose terminal verdict was already sent as FAILED: frozen
        # for the request lifetime so a later chunk can never send SUCCESS.
        self._terminal_failed: set[str] = set()
        self.k_buffer = k_buffer
        self.v_buffer = v_buffer
        self.enable_kv_quant = enable_kv_quant
        self.enable_c8_quant = enable_c8_quant
        self.ready_event = ready_event
        self.callback_func = callback_func
        self.reuse_completion_callback = reuse_completion_callback
        # Receipt mode: set by the worker for receipt-enabled engines.
        self.layer_result_sender: Callable | None = None

    def run(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        self.ready_event.set()
        while True:
            send_task = self.send_queue.get()
            if isinstance(send_task, SourceWriteJob):
                try:
                    send_task.execute(self.engine)
                except Exception:
                    logger.exception("Exact source job failed; source and targets remain retained")
                finally:
                    self.send_queue.task_done()
            else:
                self._handle_request(send_task)

    def _handle_request(self, send_task: SendTask):
        error: str | None = None
        try:
            error = self._transfer_kv_cache(send_task)
        except Exception as e:
            error = str(e)
            self.failed_reqs.update(send_task.send_request)
        finally:
            if error is not None:
                logger.error(
                    "Failed to transfer KV cache. layer_idx=%s, error=%s. Check transfer engine and memory state.",
                    send_task.layer_idx,
                    error,
                )
            self.failed_reqs.update(send_task.failed_requests)
            if send_task.layer_idx == (self.total_layers - 1):
                completion_requests = {
                    **send_task.send_request,
                    **send_task.failed_requests,
                }
                if completion_requests:
                    primary_group_idx = self.layer_metadata[send_task.layer_name].tensor_group_idx[0]
                    for req_id, req_meta in completion_requests.items():
                        if not req_meta.chunk_finish:
                            continue
                        transfer_succeeded = req_id not in self.failed_reqs and req_id not in self._terminal_failed
                        self.callback_func(
                            req_id,
                            req_meta,
                            primary_group_idx,
                            trans_flag=transfer_succeeded,
                        )
                        if not transfer_succeeded:
                            # The verdict is frozen: failed_reqs may clear for
                            # a later chunk, but a request that ever sent
                            # FAILED can never send SUCCESS afterwards.
                            self._terminal_failed.add(req_id)
                            self.failed_reqs.discard(req_id)
            if send_task.source_completion is not None:
                send_task.source_completion(RuntimeError(error) if error else None, error is None)
            if self.reuse_completion_callback is not None:
                self.reuse_completion_callback(send_task.layer_idx, error)
            self._emit_layer_results(send_task, error)
            self.send_queue.task_done()

    def _emit_layer_results(self, send_task: SendTask, error: str | None) -> None:
        """Receipt protocol sender: one LAYER_RESULT per (component,
        chunk span) per request, plus RANK_TERMINAL after the last layer (empty
        payloads included). Only receipt-mode requests (those
        carrying request_instance) participate; plain mode keeps DONE-only."""
        if self.layer_result_sender is None:
            return
        completion = {**send_task.send_request, **send_task.failed_requests}
        if not completion:
            return
        is_last_layer = send_task.layer_idx == (self.total_layers - 1)
        # Component identity comes from the registered cache-spec names — the
        # same rule register_kv_caches uses ("indexer" names are INDEXER,
        # everything else is MAIN); never from tensor counts or group ids,
        # which cannot distinguish components sharing one manager group.
        layer_names = send_task.layer_names or [send_task.layer_name]
        components: list[Component] = []
        for name in layer_names:
            component = Component.INDEXER if "indexer" in name.lower() else Component.MAIN
            if component not in components:
                components.append(component)
        for req_id, req_meta in completion.items():
            if not req_meta.request_instance:
                continue
            succeeded = error is None and req_id not in self.failed_reqs
            status = Status.SUCCESS if succeeded else Status.FAILED
            end_token = min(req_meta.local_computed_tokens or req_meta.prompt_len, req_meta.prompt_len)
            start_token = min(req_meta.local_transed_tokens, end_token)
            for component in components:
                self.layer_result_sender(
                    req_meta,
                    (
                        LAYER_RESULT_MSG,
                        CONTRACT_VERSION,
                        req_meta.request_instance,
                        0,  # transfer_generation: bumped only on preempt/retry; P learns it via the transfer manifest
                        self.tp_rank,
                        send_task.layer_idx,
                        send_task.layer_idx,
                        component.value,
                        start_token,
                        end_token,
                        "",  # backing_id: D resolves from its own lease (v1 single-backing)
                        0,
                        status.value,
                        # source quiesced = True once the send thread finished
                        # processing the task: ALL source reads for this layer
                        # are complete, for succeeded AND failed components.
                        # Failed sources are held only when a retry
                        # could re-read them; v1 has no retry path, so holding
                        # a failed component's source would just stall the
                        # slot until abort. If a retry policy is added later, switch
                        # to per-component holds (failure keeps False).
                        True,
                    ),
                )
            if is_last_layer and req_meta.chunk_finish:
                terminal_status = Status.FAILED if req_id in self._terminal_failed else status
                self.layer_result_sender(
                    req_meta,
                    (
                        RANK_TERMINAL_MSG,
                        CONTRACT_VERSION,
                        req_meta.request_instance,
                        0,
                        self.tp_rank,
                        terminal_status.value,
                    ),
                )

    def get_transfer_meta(
        self,
        send_task: SendTask,
        req_id: str,
        req_meta: ReqMeta,
        layer_name: str,
        layer_group_idx: int,
    ):
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []

        layer_kv_cache_spec = self.kv_cache_specs[layer_group_idx]
        remote_block_ids = req_meta.remote_block_ids[layer_group_idx]
        remote_layer_metadata = req_meta.remote_layer_metadata[layer_name]
        local_layer_metadata = self.layer_metadata[layer_name]
        local_block_ids = req_meta.local_block_ids[layer_group_idx]

        if isinstance(layer_kv_cache_spec, MambaSpec):
            # only support one block transfer for mamba
            if self.mamba_cache_mode == "align":
                local_transfer_idx = len(local_block_ids) - self.num_speculative_tokens - 1
            else:
                local_transfer_idx = 0
            remote_transfer_idx = len(remote_block_ids) - self.num_speculative_tokens - 1
            local_conv_addr, local_ssm_addr = local_layer_metadata.kv_caches_base_addr
            remote_conv_addr, remote_ssm_addr = remote_layer_metadata.kv_caches_base_addr
            local_conv_len, local_ssm_len = local_layer_metadata.block_len
            tp_ratio = self.tp_size // req_meta.remote_tp_size
            if tp_ratio == 1:
                src_list.extend(
                    [
                        local_conv_addr + local_block_ids[local_transfer_idx] * local_conv_len,
                        local_ssm_addr + local_block_ids[local_transfer_idx] * local_ssm_len,
                    ]
                )
                dst_list.extend(
                    [
                        remote_conv_addr + remote_block_ids[remote_transfer_idx] * local_conv_len,
                        remote_ssm_addr + remote_block_ids[remote_transfer_idx] * local_ssm_len,
                    ]
                )
                length_list.extend([local_conv_len, local_ssm_len])
            else:
                conv_shape, ssm_shape = layer_kv_cache_spec.shapes
                conv_dtype, ssm_dtype = layer_kv_cache_spec.dtypes
                remote_conv_len, remote_ssm_len = remote_layer_metadata.block_len
                # conv
                linear_key_head_dim = self.vllm_config.model_config.hf_text_config.linear_key_head_dim
                linear_num_key_heads = self.vllm_config.model_config.hf_text_config.linear_num_key_heads
                linear_value_head_dim = self.vllm_config.model_config.hf_text_config.linear_value_head_dim
                linear_num_value_heads = self.vllm_config.model_config.hf_text_config.linear_num_value_heads
                local_num_key_heads = linear_num_key_heads // self.tp_size
                local_num_value_heads = linear_num_value_heads // self.tp_size
                local_conv_offsets = [
                    0,
                    local_num_key_heads * linear_key_head_dim,
                    local_num_key_heads * 2 * linear_key_head_dim,
                ]
                local_conv_sizes = [
                    local_num_key_heads * linear_key_head_dim,
                    local_num_key_heads * linear_key_head_dim,
                    local_num_value_heads * linear_value_head_dim,
                ]
                for i in range(conv_shape[0]):
                    for local_conv_offset, local_conv_size in zip(local_conv_offsets, local_conv_sizes):
                        local_addr_offset = (i * conv_shape[1] + local_conv_offset) * get_dtype_size(conv_dtype)
                        remote_addr_offset = (
                            (i * conv_shape[1] + local_conv_offset) * tp_ratio
                            + (self.tp_rank % tp_ratio) * local_conv_size
                        ) * get_dtype_size(conv_dtype)
                        src_list.append(
                            local_conv_addr + local_block_ids[local_transfer_idx] * local_conv_len + local_addr_offset
                        )
                        dst_list.append(
                            remote_conv_addr
                            + remote_block_ids[remote_transfer_idx] * remote_conv_len
                            + remote_addr_offset
                        )
                        length_list.append(local_conv_size * get_dtype_size(conv_dtype))
                # ssm
                remote_addr_offset = (self.tp_rank % tp_ratio) * math.prod(ssm_shape) * get_dtype_size(ssm_dtype)
                src_list.append(local_ssm_addr + local_block_ids[local_transfer_idx] * local_ssm_len)
                dst_list.append(
                    remote_ssm_addr + remote_block_ids[remote_transfer_idx] * remote_ssm_len + remote_addr_offset
                )
                length_list.append(local_ssm_len)
        else:
            use_staging_buffer = layer_name == send_task.layer_name and (
                send_task.k_cache is not None or send_task.k_quant_cache is not None
            )
            if self.pd_head_ratio == 1 or not use_staging_buffer:
                layer_local_kv_base_addr = local_layer_metadata.kv_caches_base_addr
                layer_remote_kv_base_addr = remote_layer_metadata.kv_caches_base_addr
                block_lens = local_layer_metadata.block_len
                grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
                    remote_block_ids, local_block_ids
                )
                # kv cache quantization scenario
                if (
                    use_staging_buffer
                    and (self.enable_kv_quant or self.enable_c8_quant)
                    and send_task.k_quant_cache is not None
                ):
                    assert len(block_lens) == 2, "Quantization block length must be 2!"
                    if self.enable_kv_quant:
                        quant_block_lens = [block_lens[0] // 2, block_lens[1]]
                    else:
                        quant_block_lens = [block_lens[0] // 2, block_lens[1] // 2]
                    layer_local_quant_kv_addr = [self.k_buffer.data_ptr(), self.v_buffer.data_ptr()]
                    rearrange_block_ids = send_task.group_rearrange_block_ids[layer_group_idx]
                    # eg:[5,6,7,9] -> {5:0, 6:1, 7:2, 9:3}
                    rearrange_block_dict = {
                        value: index
                        for index, value in enumerate(rearrange_block_ids)  # type:ignore
                    }
                    for block_len, src_layer_base_addr, dst_layer_base_addr in zip(
                        quant_block_lens, layer_local_quant_kv_addr, layer_remote_kv_base_addr
                    ):
                        for group_remote_block_id, group_local_block_id in zip(
                            grouped_remote_block_ids, grouped_local_block_ids
                        ):
                            src = src_layer_base_addr + rearrange_block_dict[group_local_block_id[0]] * block_len
                            dst = dst_layer_base_addr + group_remote_block_id[0] * block_len
                            length = len(group_local_block_id) * block_len
                            src_list.append(src)
                            dst_list.append(dst)
                            length_list.append(length)
                else:
                    for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                        zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)
                    ):
                        block_len = block_lens[k]
                        for group_remote_block_id, group_local_block_id in zip(
                            grouped_remote_block_ids, grouped_local_block_ids
                        ):
                            src = src_layer_base_addr + group_local_block_id[0] * block_len
                            dst = dst_layer_base_addr + group_remote_block_id[0] * block_len
                            length = len(group_local_block_id) * block_len
                            src_list.append(src)
                            dst_list.append(dst)
                            length_list.append(length)
            else:
                rearrange_block_ids = send_task.group_rearrange_block_ids[layer_group_idx]
                rearrange_block_dict = {
                    value: index
                    for index, value in enumerate(rearrange_block_ids)  # type:ignore
                }
                layer_local_kv_base_addr = [self.k_buffer.data_ptr(), self.v_buffer.data_ptr()]
                layer_remote_kv_base_addr = remote_layer_metadata.kv_caches_base_addr
                block_lens = local_layer_metadata.block_len
                remote_block_lens = remote_layer_metadata.block_len
                assert len(layer_remote_kv_base_addr) == 2, (
                    "Layer kv_cache resharding only supports two kv cache tensors."
                )
                src_list, dst_list, length_list = [], [], []
                for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                    zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)
                ):
                    block_len = block_lens[k]
                    if self.enable_c8_quant:
                        block_len = block_len // 2
                    remote_block_len = remote_block_lens[k]
                    for remote_block_id, local_block_id in zip(remote_block_ids, local_block_ids):
                        src = src_layer_base_addr + rearrange_block_dict[local_block_id] * block_len
                        dst = (
                            dst_layer_base_addr
                            + remote_block_id * remote_block_len
                            + block_len * ((self.tp_rank // self.num_head_replica) % self.pd_head_ratio)
                        )
                        src_list.append(src)
                        dst_list.append(dst)
                        length_list.append(block_len)
        return (src_list, dst_list, length_list)

    def _transfer_kv_cache(self, send_task: SendTask) -> str | None:
        if send_task.source_ready_event is not None:
            send_task.source_ready_event.synchronize()
        key = send_task.k_cache
        value = send_task.v_cache
        if self.pd_head_ratio > 1 and key is not None and value is not None:
            with npu_stream_switch(self.resharding_stream):
                key = key.view(-1, key.shape[-1])  # type:ignore
                value = value.view(-1, key.shape[-1])  # type:ignore
                self.k_buffer[: key.shape[0]].copy_(key)  # [:4, 128] ->
                self.v_buffer[: value.shape[0]].copy_(value)
        if send_task.k_quant_cache is not None:
            with npu_stream_switch(self.resharding_stream):
                key_quant = send_task.k_quant_cache
                key_quant = key_quant.view(-1, key_quant.shape[-1])  # type:ignore
                self.k_buffer[: key_quant.shape[0]].copy_(key_quant)
                value_quant = send_task.v_quant_cache
                value_quant = value_quant.view(-1, value_quant.shape[-1])  # type:ignore
                self.v_buffer[: value_quant.shape[0]].copy_(value_quant)

        # Merge transmission tasks of the same session
        session_meta: dict[str, TransferMeta] = {}
        layer_names = send_task.layer_names or [send_task.layer_name]
        layer_group_idx = self.layer_metadata[send_task.layer_name].tensor_group_idx[0]
        for req_id, req_meta in send_task.send_request.items():
            # Prefer the target lease's transport endpoint (carried by the
            # receipt lease's RegisteredRegion); fall back to the zmq-handshake
            # host:port for plain mode or a lease that did not carry it.
            session_id = req_meta.remote_te_rpc_endpoint or (
                f"{req_meta.remote_host}:{req_meta.remote_te_rpc_port}"
                if req_meta.remote_host and req_meta.remote_te_rpc_port is not None
                else None
            )
            if session_id is None:
                raise RuntimeError(
                    f"Cannot open a mooncake session for {req_id}: no target endpoint and no handshake port"
                )
            if session_id not in session_meta:
                session_meta[session_id] = TransferMeta(src=[], dst=[], length=[], req_ids=[])

            for layer_name in layer_names:
                src_list, dst_list, length_list = self.get_transfer_meta(
                    send_task,
                    req_id,
                    req_meta,
                    layer_name,
                    layer_group_idx,
                )
                session_meta[session_id].src.extend(src_list)
                session_meta[session_id].dst.extend(dst_list)
                session_meta[session_id].length.extend(length_list)
            session_meta[session_id].req_ids.append(req_id)

        if send_task.k_quant_cache is not None:
            self.resharding_stream.synchronize()
        elif send_task.k_cache is None or self.pd_head_ratio == 1:
            """
            Note: Due to a bug in ADXL, calling current_event.synchronize() may occasionally hang.
            This issue will be fixed in CANN version 8.5.rc1.
            You can manually build the master branch of the project at https://gitcode.com/cann/hixl
            to resolve this issue before the 8.5.RC1 release.
            """
            send_task.wait_event.synchronize()  # type:ignore
        elif self.pd_head_ratio > 1:
            self.resharding_stream.synchronize()

        transfer_errors: list[str] = []
        for session_id, transfer_meta in session_meta.items():
            if len(transfer_meta.src) > 0:
                req_start_time = time.perf_counter()
                ret = self.engine.batch_transfer_sync_write(
                    session_id, transfer_meta.src, transfer_meta.dst, transfer_meta.length
                )
                if ret < 0:
                    self.failed_reqs.update(transfer_meta.req_ids)
                    transfer_errors.append(
                        "Mooncake transfer failed for send requests. "
                        f"req_ids={transfer_meta.req_ids}, destination={session_id}, ret={ret}"
                    )
                else:
                    req_end_time = time.perf_counter()
                    total_transfer_size = sum(transfer_meta.length) / 1024
                    req_transfer_elapsed = (req_end_time - req_start_time) * 1000
                    logger.debug(
                        "Layer%d KV cache transfer task %dKB to remote_session_id [%s] took %.3f ms.",
                        send_task.layer_idx,
                        total_transfer_size,
                        session_id,
                        req_transfer_elapsed,
                    )
        return "; ".join(transfer_errors) if transfer_errors else None


class KVCacheRecvingLayerThread(threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        side_channel_port: int,
        tp_size: int,
        pd_head_ratio: int,
        local_engine_id: str,
        metadata: MooncakeAgentMetadata,
        ready_event: threading.Event,
    ):
        super().__init__(daemon=True, name="KVCacheRecvingLayerThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pd_head_ratio = pd_head_ratio
        self.local_engine_id = local_engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = side_channel_port
        self.lock = threading.Lock()
        self.done_requests = set[str]()
        self.failed_requests = set[str]()
        self.task_tracker = dict[str, int]()
        self.ready_event = ready_event
        self.metadata = metadata
        # Receipt-mode state one ReceiptLedger per
        # in-flight request, published by the scheduler side after target
        # acquisition. This receive thread is the sole ledger writer; keyed by
        # request_instance (what P stamps into receipt messages).
        self.ledger_by_req: dict[str, ReceiptLedger] = {}
        self.instance_to_req: dict[str, str] = {}
        # Request ids ever seen in receipt mode; kept for the request lifetime
        # (not popped on publish) so a late versioned DONE after publish is
        # recognized as normal instead of flagged as a mode mismatch.
        self.receipt_mode_reqs: set[str] = set()
        self.receipt_provider = None  # TargetProvider, set via the facade
        # instance -> monotonic publish deadline, set at ledger install and
        # swept once per second by the receive-thread loop (B5 watchdog).
        self.receipt_deadlines: dict[str, float] = {}

    def get_and_clear_done_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.lock:
            finished_requests = self.done_requests
            self.done_requests = set()
        return finished_requests

    def get_and_clear_failed_requests(self) -> set[str]:
        """
        Get and clear the requests that have failed.
        Returns:
            A set of request IDs that have failed.
        """
        with self.lock:
            failed_requests = self.failed_requests
            self.failed_requests = set()
        return failed_requests

    def update_failed_task(self, req_id: str) -> None:
        """
        Handle a failed task by adding it to the failed_requests set and removing it from the task tracker.
        Args:
            req_id: The ID of the request that has failed.
        """
        with self.lock:
            if req_id not in self.task_tracker:
                self.task_tracker[req_id] = set()
            self.task_tracker.pop(req_id, None)
            self.failed_requests.add(req_id)

    def update_done_task(self, req_id, trans_count, side_channel_path):
        """
        Handle a completed task by adding it to the done_requests set and removing it from the task tracker.
        Args:
            req_id: The ID of the request that has completed.
        """
        with self.lock:
            if req_id not in self.task_tracker:
                self.task_tracker[req_id] = set()
            self.task_tracker[req_id].add(side_channel_path)
            if len(self.task_tracker[req_id]) == trans_count:
                self.task_tracker.pop(req_id)
                self.done_requests.add(req_id)

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
            ledger = self.ledger_by_req.get(identity.request_instance)
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
            logger.error("Malformed LAYER_RESULT_MSG (expected 14 fields): %r", msg)
            return
        if type(version) is not int or version != CONTRACT_VERSION:
            logger.error("LAYER_RESULT_MSG version mismatch: %s != %s", version, CONTRACT_VERSION)
            return
        with self.lock:
            ledger = self.ledger_by_req.get(instance)
        if ledger is None:
            logger.debug("LAYER_RESULT for unknown instance %s (plain-mode or stale); dropped", instance)
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
            logger.error(
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
            logger.error("Invalid LAYER_RESULT_MSG fields for %s: %s (msg=%r)", instance, e, msg)
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
            logger.error("Malformed RANK_TERMINAL_MSG (expected 6 fields): %r", msg)
            return
        if type(version) is not int or version != CONTRACT_VERSION:
            logger.error("RANK_TERMINAL_MSG version mismatch: %s != %s", version, CONTRACT_VERSION)
            return
        with self.lock:
            ledger = self.ledger_by_req.get(instance)
        if ledger is None:
            logger.warning("RANK_TERMINAL for unknown instance %s; dropped", instance)
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
            logger.error("Invalid rank terminal for %s: %s", instance, error)
            return
        self._maybe_publish(instance)

    def _maybe_publish(self, instance: str) -> None:
        """Visibility flow, run in the receive thread (sole writer):
        data ready -> receipt(visible=False) -> provider.prepare_decode -> normal
        return -> mark visible -> done_requests. Any prepare_decode error means
        never visible, never published."""
        with self.lock:
            ledger = self.ledger_by_req.get(instance)
            req_id = self.instance_to_req.get(instance)
            if ledger is None or req_id is None or req_id in self.done_requests:
                return
        receipt = ledger.build_receipt()
        if isinstance(ledger, ExactReceiptLedger):
            if ledger.delivery_aborted or ledger.should_publish():
                return
            if ledger.failed:
                ledger.abort_delivery()
                with self.lock:
                    self.failed_requests.add(req_id)
                    self.receipt_deadlines.pop(instance, None)
                return
        if receipt is None:
            return  # data conditions not met yet
        provider = self.receipt_provider
        lease = getattr(ledger, "lease", None)
        if provider is None or lease is None:
            # A receipt-mode request can never fall back to plain-mode
            # publishing: without prepare_decode the consumer dependency is
            # never wired and the target bytes are not safe to decode. Fail
            # the request through the bounded-failure channel instead of
            # silently releasing it as done.
            logger.error(
                "Receipt-mode request %s (instance=%s) has no provider or lease; failing instead of publishing",
                req_id,
                instance,
            )
            with self.lock:
                self.failed_requests.add(req_id)
                self.receipt_deadlines.pop(instance, None)
            return
        try:
            if isinstance(ledger, ExactReceiptLedger):
                provider.confirm_native_received(lease, receipt)
            provider.prepare_decode(lease, receipt)
        except Exception as e:
            logger.error(
                "prepare_decode failed for instance %s (req %s); not publishing. error=%s", instance, req_id, e
            )
            return
        ranks = (ledger.visibility_rank,) if isinstance(ledger, ExactReceiptLedger) else sorted(ledger.expected_ranks)
        for rank in ranks:
            ledger.mark_visible(rank)
        receipt.visible = True
        with self.lock:
            # Keep receipt_mode_reqs (request-lifetime): late versioned DONE
            # after publish must stay recognized as normal.
            self.receipt_mode_reqs.add(req_id)
            if not isinstance(ledger, ExactReceiptLedger):
                self.ledger_by_req.pop(instance, None)
                self.instance_to_req.pop(instance, None)
            self.receipt_deadlines.pop(instance, None)
            self.done_requests.add(req_id)
        logger.info("Receipt-mode request %s (instance=%s) published after prepare_decode", req_id, instance)

    def forget_request(self, req_id: str, *, failed=False) -> None:
        """Clean legacy receives; retain exact proof identity until owner retirement."""
        with self.lock:
            stale = [i for i, r in self.instance_to_req.items() if r == req_id]
            for instance in stale:
                ledger = self.ledger_by_req.get(instance)
                if isinstance(ledger, ExactReceiptLedger):
                    if failed:
                        ledger.abort_delivery()
                    continue  # Keep late quiescence proof until actual owner retirement.
                self.instance_to_req.pop(instance, None)
                self.ledger_by_req.pop(instance, None)
                self.receipt_deadlines.pop(instance, None)
            if req_id not in self.instance_to_req.values():
                self.receipt_mode_reqs.discard(req_id)

    def _sweep_receipt_deadlines(self) -> None:
        """Bounded-failure watchdog sweep (B5): fail receipt requests whose
        ledger was installed but never published within the deadline. Exact
        ledgers continue accepting late quiescence proofs, without decoding."""
        now = time.monotonic()
        expired: list[tuple[str, str]] = []
        with self.lock:
            for instance, deadline in self.receipt_deadlines.items():
                req_id = self.instance_to_req.get(instance)
                if now >= deadline and req_id is not None:
                    expired.append((instance, req_id))
            for instance, req_id in expired:
                ledger = self.ledger_by_req.get(instance)
                if isinstance(ledger, ExactReceiptLedger):
                    ledger.abort_delivery()
                else:
                    self.ledger_by_req.pop(instance, None)
                    self.instance_to_req.pop(instance, None)
                self.receipt_deadlines.pop(instance, None)
                self.failed_requests.add(req_id)
        for instance, req_id in expired:
            logger.error(
                "Receipt watchdog: request %s (instance=%s) did not publish within %.0fs; failing bounded",
                req_id,
                instance,
                RECEIPT_WATCHDOG_SECONDS,
            )

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        handshake_port = self.side_channel_port + self.tp_rank
        path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
        logger.info("KVCacheRecvingLayerThread listening on %s, tp_rank=%d", path, self.tp_rank)
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
            self.ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=tuple)
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            while True:
                try:
                    # A 1s poll keeps the receipt-watchdog sweep running even
                    # when P goes completely silent.
                    if not dict(poller.poll(timeout=1000)):
                        self._sweep_receipt_deadlines()
                        continue
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        logger.error(
                            "Invalid message format. expected>=2 frames, got %d. frames=%s", len(frames), frames
                        )
                        continue

                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        logger.error("Invalid payload count. expected=1, got %d. frames=%s", len(payload), frames)
                        continue

                    msg = decoder.decode(payload[0])
                    if msg[0] == GET_META_MSG:
                        logger.info("Got GET META INFO for request %s", msg[0])
                        sock.send_multipart((identity, b"", encoded_data))
                    elif msg[0] == GET_TARGET_REGIONS_MSG:
                        try:
                            if len(msg) != 2 or self.receipt_provider is None:
                                raise ContractError("Target region lookup has no registered owner")
                            lease = target_lease_from_wire(msg[1])
                            delivery = getattr(self, "target_delivery", None)
                            if delivery is None:
                                raise ContractError("Target region lookup has no admitted writer gate")
                            with delivery.source_lookup(lease):
                                handles = self.receipt_provider.resolve_target_regions(lease)
                            response = encode_target_addresses(lease, handles)
                        except Exception as error:
                            response = (TARGET_PEER_ERROR, type(error).__name__, str(error))
                        sock.send_multipart((identity, b"", encoder.encode(response)))
                    elif msg[0] == DONE_SENDING_MSG:
                        logger.debug("Got DONE_RECVING_MSG for request %s", msg[1])
                        request_id = msg[1]
                        trans_count = msg[2]
                        side_channel_path = msg[3]
                        if len(msg) >= 5 and msg[4] == CONTRACT_VERSION:
                            # Receipt-mode request, marked by the version field
                            # P appends for requests carrying request_instance.
                            # Request-intrinsic fact: no dependency on when the
                            # scheduler synced the ledger maps. Plain
                            # requests keep the 4-tuple and reach
                            # update_done_task unchanged.
                            if request_id in self.receipt_mode_reqs:
                                logger.debug("Receipt-mode request %s: DONE ignored, ledger owns publish", request_id)
                            else:
                                # Diagnostic only, never gates: P has receipt
                                # mode on but this engine has no ledger for the
                                # request (mode mismatch across P/D, or the
                                # sync was lost). Log loudly so the watchdog
                                # turns it into a BOUNDED failure with a clear
                                # cause instead of a silent hang.
                                logger.error(
                                    "Versioned DONE for request %s but no receipt-mode state on this engine "
                                    "(check layerwise_receipt_mode on P and D, and ledger sync); "
                                    "publish would hang until the watchdog bounds it",
                                    request_id,
                                )
                        else:
                            self.update_done_task(request_id, trans_count, side_channel_path)
                        sock.send_multipart((identity, b"", b"ACK"))
                    elif msg[0] == FAILED_SENDING_MSG:
                        request_id = msg[1]
                        logger.error("Got FAILED_SENDING_MSG for request. request_id=%s. ", msg[1])
                        self.update_failed_task(request_id)
                        sock.send_multipart((identity, b"", b"ACK"))
                    elif msg[0] == LAYER_RESULT_MSG:
                        self.handle_layer_result(msg)
                        sock.send_multipart((identity, b"", b"ACK"))
                    elif msg[0] == RANK_TERMINAL_MSG:
                        self.handle_rank_terminal(msg)
                        sock.send_multipart((identity, b"", b"ACK"))
                    elif msg[0] == LAYER_RESULTS_MSG:
                        sock.send_multipart((identity, b"", self.handle_proof_batch(msg)))
                    else:
                        logger.error(
                            "Unexpected message type: %s. expected GET_META_MSG or DONE_RECVING_MSG. msg=%s",
                            msg[0] if msg else "empty",
                            msg,
                        )
                    self._sweep_receipt_deadlines()
                except Exception as e:
                    logger.error(
                        "Failed to decode message. type=%s, error=%s. context=decoding payload", type(e).__name__, e
                    )


class MooncakeLayerwiseConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.send_task: SendTask = SendTask()
        # Older metadata can carry receipt specs. Current receipt targets
        # install their ledgers synchronously over the owner RPC.
        self.receipt_specs: dict[str, tuple] = {}
        self.scheduled_target_allocations: dict[str, BlockAllocation] | None = None

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        token_ids: list[int] | None = None,
        chunk_finish: bool = False,
        prompt_len: int = 0,
        remote_cache_tokens: int = 0,
        local_computed_tokens: int = 0,
        local_transed_tokens: int = 0,
        remote_target: tuple[TargetLease | None, TargetManifest | None] | None = None,
    ):
        if remote_target is None:
            wire_lease = kv_transfer_params.get("remote_target_lease")
            lease = target_lease_from_wire(wire_lease) if wire_lease is not None else None
            wire_manifest = kv_transfer_params.get("remote_target_manifest")
            manifest = target_manifest_from_wire(wire_manifest) if wire_manifest is not None else None
            if manifest is not None and manifest.lease != lease:
                raise ContractError("Published manifest and aggregate lease do not identify the same target")
        else:
            lease, manifest = remote_target
        instance = kv_transfer_params.get("request_instance", "")
        if lease is not None and lease.request.request_instance != instance:
            raise ContractError("Published target lease belongs to another request instance")
        if instance and kv_transfer_params.get("do_remote_decode") and lease is None:
            raise ContractError("Receipt producer request has no published target lease")
        self.requests[request_id] = ReqMeta(
            token_ids=token_ids or [],
            local_block_ids=local_block_ids,
            request_instance=kv_transfer_params.get("request_instance", ""),
            remote_block_ids=kv_transfer_params.get("remote_block_ids", []),
            remote_block_size=kv_transfer_params.get("remote_block_size", []),
            remote_engine_id=kv_transfer_params.get("remote_engine_id"),
            remote_host=kv_transfer_params.get("remote_host"),
            remote_port=kv_transfer_params.get("remote_port"),
            remote_te_rpc_port=kv_transfer_params.get("remote_te_rpc_port"),
            remote_te_rpc_endpoint=kv_transfer_params.get("remote_te_rpc_endpoint"),
            remote_writer_plan=lease.writer_plan if lease is not None else None,
            remote_target_lease=lease,
            remote_target_manifest=manifest,
            remote_layer_metadata=kv_transfer_params.get("remote_layer_metadata"),
            metaserver=kv_transfer_params.get("metaserver"),
            remote_tp_size=kv_transfer_params.get("remote_tp_size"),
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size"),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size"),
            do_virtual=kv_transfer_params.get("do_virtual"),
            chunk_finish=chunk_finish,
            remote_cache_tokens=remote_cache_tokens,
            local_computed_tokens=local_computed_tokens,
            prompt_len=prompt_len,
            local_transed_tokens=local_transed_tokens,
            restore_only=bool(kv_transfer_params.get("restore_only", False)),
            restore_end_token=int(kv_transfer_params.get("restore_end_token", 0)),
            token_len_chunk=int(kv_transfer_params.get("token_len_chunk", 0)),
            trans_count=[],
        )


class MooncakeLayerwiseConnector(KVConnectorBase_V1, SupportsHMA):
    requires_full_blocks_on_update_after_alloc = True
    supports_layerwise_buffer_reuse = True
    reuse_source_contract_version = 1

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.is_producer = self._is_kv_producer
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeLayerwiseConnectorMetadata()
        self._target_provider = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeLayerwiseConnectorScheduler | None = MooncakeLayerwiseConnectorScheduler(
                vllm_config, kv_cache_config, str(self.engine_id)
            )
            self.connector_worker: MooncakeLayerwiseConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeLayerwiseConnectorWorker(vllm_config, kv_cache_config, str(self.engine_id))

        self.target_provider = None
        self._target_owner_binding = None
        self._target_owner_dispatcher = None
        self._worker_target_rpc = None
        self._scheduler_target_rpc = None
        self._target_rpc_config = None
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        requested = extra.get("layerwise_receipt_mode", False)
        if type(requested) is not bool:
            raise ContractError("layerwise_receipt_mode must be a boolean")
        if requested and vllm_config.kv_transfer_config.is_kv_consumer:
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
                DeferredTargetProvider,
                TargetProviderClient,
            )
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_factory import TargetOwnerRpcConfig

            if role == KVConnectorRole.SCHEDULER:
                self.target_provider = TargetProviderClient()
            else:
                self.target_provider = DeferredTargetProvider()
                self.connector_worker.target_provider = self.target_provider
            self._target_rpc_config = TargetOwnerRpcConfig.from_extra(
                extra, vllm_config.parallel_config.tensor_parallel_size
            )
            self.set_target_provider(self.target_provider)

    @property
    def requires_target_owner(self) -> bool:
        return self.connector_worker is not None and self.target_provider is not None

    def configure_target_owner(self, manager, *, retention, consumption_stream, timeout: float, te=None) -> None:
        """A2 worker-local initialization, before register_kv_caches.

        B supplies its real retention client and uses the same TE registration
        owner as the connector. The actual catalog is constructed AFTER that
        registration. No caller may serialize these local objects.

        When retention is omitted, explicit loopback RPC settings select the
        reverse scheduler authority. Construction does not wait for scheduler
        initialization. Tests/deployment adapters can inject an authority.
        """
        if self.connector_worker is None or self.target_provider is None:
            raise ContractError("Target owner binding is only available on receipt Decode workers")
        if self._target_owner_binding is not None or self.target_provider.ready:
            raise ContractError("Target owner initialization cannot be replaced")
        te = global_te if te is None else te
        if te.transfer_engine is not self.connector_worker.engine:
            raise ContractError("Target catalog must use this connector's actual TE registration owner")
        if retention is None:
            if self._target_rpc_config is None:
                raise ContractError("Receipt workers require explicit RPC ports or an injected retention authority")
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_factory import WorkerTargetRpc

            self._worker_target_rpc = WorkerTargetRpc(
                self._target_rpc_config, self.connector_worker.tp_rank, self.target_provider
            )
            retention = self._worker_target_rpc.retention
        self._target_owner_binding = (manager, te, retention, consumption_stream, timeout)
        self.connector_worker.target_owner_manager = manager

    def bind_target_owner_transport(self, transport=None) -> None:
        """A2 scheduler facade; B binds its actual worker-owner channel here.

        Explicit RPC settings connect every TP worker; an injected transport
        remains available for focused adapters/tests. No global rendezvous is
        used by this initialization path.
        """
        if self.connector_scheduler is None or self.target_provider is None:
            raise ContractError("Target owner transport is only available on receipt Decode schedulers")
        if transport is None:
            if self._target_rpc_config is None:
                raise ContractError("Receipt schedulers require explicit RPC ports or an injected owner transport")
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_factory import SchedulerTargetRpc

            self._scheduler_target_rpc = SchedulerTargetRpc(
                self._target_rpc_config, self.connector_scheduler._retention_registry
            )
            transport = self._scheduler_target_rpc.transport
        self.target_provider.bind_transport(transport)

    def configure_receipt_scheduler_runtime(self, manager, allocation_lock):
        if self.connector_scheduler is None or self.target_provider is None:
            return None
        self.connector_scheduler.bind_scheduler_retention(manager, allocation_lock=allocation_lock)
        registry = self.connector_scheduler._retention_registry
        manager.bind_retention_registry(registry)
        self.bind_target_owner_transport()
        return registry

    def prepare_target_free(self, request) -> bool:
        if self.connector_scheduler is None:
            return True
        return self.connector_scheduler._release_target_if_any(request)

    def is_target_ready(self, request) -> bool:
        return self.connector_scheduler is None or self.connector_scheduler._is_target_ready(request)

    def target_use_context(self, request_ids, metadata=None):
        delivery = getattr(self.connector_worker, "_target_delivery", None)
        if delivery is None:
            if self.target_provider is not None and self.target_provider.ready:
                raise ContractError("Registered target owner has no model-use admission registry")
            return contextlib.nullcontext()  # Allocation/profile phase has no admitted target.
        allocations = getattr(metadata, "scheduled_target_allocations", None)
        return delivery.use_context(request_ids, allocations)

    def shutdown(self) -> None:
        writer = getattr(self.connector_worker, "_manifest_source_writer", None)
        if writer is not None:
            writer.close()
        for binding in (getattr(self, "_scheduler_target_rpc", None), getattr(self, "_worker_target_rpc", None)):
            if binding is not None:
                binding.close()
        super().shutdown()

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def bind_kv_cache_manager(self, kv_cache_manager) -> None:
        """Use vLLM's normal post-construction scheduler binding hook.

        MultiConnector forwards this hook to its children. It works with the
        deployed default scheduler as well as custom scheduler subclasses.
        """
        super().bind_kv_cache_manager(kv_cache_manager)
        if self.connector_scheduler is not None and self.target_provider is not None:
            self.connector_scheduler.bind_scheduler_retention(kv_cache_manager)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def update_state_before_preempt(self, request: "Request"):
        """Preempt notification only. Receipt-mode retention stays: the
        request may resume, so the target lease and the scheduler retention
        record are kept and blocks must not return to the pool here."""
        scheduler = self.connector_scheduler
        if scheduler is None:
            return
        instance = (request.kv_transfer_params or {}).get("request_instance")
        if instance:
            logger.info(
                "Receipt-mode request %s preempted (instance=%s); target lease and retention kept",
                request.request_id,
                instance,
            )

    def set_target_provider(self, provider) -> None:
        """The only capability-probe entry point. AscendMultiConnector
        calls this; capability gaps abort startup."""
        assert_provider_capabilities(provider)
        if getattr(self, "_target_provider", None) is provider:
            return
        if getattr(self, "_target_provider", None) is not None:
            raise ContractError("Cannot replace the target provider across owner generations")
        self._target_provider = provider
        if self.connector_scheduler is not None:
            self.connector_scheduler.attach_target_provider(provider)
        if self.connector_worker is not None:
            self.connector_worker.attach_target_provider(provider)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        facade = getattr(self, "target_provider", None)
        binding = getattr(self, "_target_owner_binding", None)
        if facade is not None and binding is None:
            raise ContractError("Configure the real target owner/retention/stream before receipt KV registration")
        self.connector_worker.register_kv_caches(kv_caches)
        if facade is not None:
            manager, te, retention, stream, timeout = binding
            manager.bind_registered_target_owner(
                kv_caches,
                te,
                retention=retention,
                consumption_stream=stream,
                timeout=timeout,
                facade=facade,
            )
            if not facade.ready:
                raise ContractError("Receipt target owner is not ready after KV registration")
            if self.connector_worker.kv_recv_layer_thread is not None:
                self.connector_worker.kv_recv_layer_thread.receipt_provider = facade
            # Publish only after real allocation/registration; independent TP
            # processes serve their explicit local endpoint, not a global slot.
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_delivery import WorkerTargetDelivery
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import TargetOwnerDispatcher

            self.connector_worker._target_delivery = WorkerTargetDelivery(
                self.connector_worker.tp_rank,
                facade,
                self.connector_worker._install_admitted_target,
                retire=self.connector_worker._retire_admitted_target,
            )
            self.connector_worker.kv_recv_layer_thread.target_delivery = self.connector_worker._target_delivery
            self._target_owner_dispatcher = TargetOwnerDispatcher(
                facade, delivery=self.connector_worker._target_delivery
            )
            worker_rpc = getattr(self, "_worker_target_rpc", None)
            if worker_rpc is not None:
                worker_rpc.serve_registered_owner(self._target_owner_dispatcher)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_transfer_results(self, finished_req_ids):
        results = super().get_transfer_results(finished_req_ids)
        failures = self.connector_worker.take_failed_receives()
        results.failed_recving.update(failures)
        results.finished_recving.update(failures)
        return results

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get the block ids that have load errors."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        assert isinstance(connector_metadata, MooncakeLayerwiseConnectorMetadata)
        assert self.connector_worker is not None
        # Newer V1 runners may defer start_load_kv until after target forward.
        # Layerwise writes need the step's counters and send mappings beforehand.
        self.connector_worker.prepare_layerwise_step(connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeLayerwiseConnector does not do layerwise saving."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.wait_for_layer_load(layer_name)

    def wait_for_layer_reuse(self, layer_idx: int) -> None:
        assert self.connector_worker is not None
        self.connector_worker.wait_for_layer_reuse(layer_idx)

    # Source consumer v1: the five callbacks are driven by the Store-side
    # registry. They delegate to the worker's adapter, which maps them onto the
    # existing slot ledger and send path so no second send path starts.
    def bind_reuse_source_contract(self) -> None:
        assert self.connector_worker is not None
        self.connector_worker._source_contract_bound = True

    def open_source(self, source) -> None:
        assert self.connector_worker is not None
        consumer = getattr(self.connector_worker, "_reuse_source_consumer", None)
        if consumer is not None:
            consumer.open_source(source)

    def source_ready(self, source, requests, ready_event) -> None:
        assert self.connector_worker is not None
        consumer = getattr(self.connector_worker, "_reuse_source_consumer", None)
        if consumer is not None:
            consumer.source_ready(source, requests, ready_event)

    def seal_source(self, source) -> None:
        assert self.connector_worker is not None
        consumer = getattr(self.connector_worker, "_reuse_source_consumer", None)
        if consumer is not None:
            consumer.seal_source(source)

    def wait_source(self, source) -> None:
        assert self.connector_worker is not None
        consumer = getattr(self.connector_worker, "_reuse_source_consumer", None)
        if consumer is not None:
            consumer.wait_source(source)

    def abort_source(self, source, error) -> None:
        assert self.connector_worker is not None
        consumer = getattr(self.connector_worker, "_reuse_source_consumer", None)
        if consumer is not None:
            consumer.abort_source(source, error)

    def on_kv_cache_written(self, layer_name: str = "") -> None:
        if not self._is_kv_producer or not self.has_connector_metadata():
            return
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        if not self._connector_metadata.requests:
            return
        assert self.connector_worker is not None
        self.connector_worker.on_kv_cache_written(layer_name)

    def save_kv_layer(
        self, layer_name: str, kv_layer: list[torch.Tensor], attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata, self._connector_metadata)

    def wait_for_save(self):
        """MooncakeLayerwiseConnector does not save explicitly."""
        pass


class MooncakeLayerwiseConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.block_size = [group_spec.kv_cache_spec.block_size for group_spec in kv_cache_config.kv_cache_groups]
        self.engine_id = engine_id
        logger.info("Initializing Mooncake Scheduler %s", engine_id)

        self.side_channel_host = get_ip()

        # disable prefill context parallel on decoder nodes
        if vllm_config.kv_transfer_config.is_kv_consumer:
            assert vllm_config.parallel_config.prefill_context_parallel_size == 1, (
                "Prefill context parallel is not support on decoder nodes"
            )

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size
        )

        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, list[int], list[list[int]]]] = {}
        self._reqs_need_send_layerwise: dict[str, SendReqInfo] = {}
        self.need_truncate = self._has_attn_mamba_hybrid_cache(kv_cache_config)
        self.executor = ThreadPoolExecutor(32)
        tls_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("tls_config", {})
        ssl_keyfile = tls_config.get("ssl_keyfile")
        ssl_certfile = tls_config.get("ssl_certfile")
        ssl_ca_certs = tls_config.get("ssl_ca_certs", False)
        ssl_keyfile_password = tls_config.get("ssl_keyfile_password")
        self.cert_path = (ssl_certfile, ssl_keyfile, ssl_keyfile_password)
        self.ssl_enable = tls_config.get("ssl_enable", False)
        self.ca_path = ssl_ca_certs
        if self.ssl_enable:
            self.metaserver_client = httpx.Client(
                limits=httpx.Limits(max_connections=100000), timeout=None, cert=self.cert_path, verify=self.ca_path
            )
        else:
            self.metaserver_client = httpx.Client(limits=httpx.Limits(max_connections=100000), timeout=None)

    @staticmethod
    def _iter_kv_cache_specs(kv_cache_config: KVCacheConfig):
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                yield from kv_cache_spec.kv_cache_specs.values()
            else:
                yield kv_cache_spec

    @classmethod
    def _has_attn_mamba_hybrid_cache(cls, kv_cache_config: KVCacheConfig) -> bool:
        has_attn = False
        has_mamba = False
        for kv_cache_spec in cls._iter_kv_cache_specs(kv_cache_config):
            has_attn = has_attn or isinstance(kv_cache_spec, AttentionSpec)
            has_mamba = has_mamba or isinstance(kv_cache_spec, MambaSpec)
        return has_attn and has_mamba

    def _hybrid_prefill_token_count(self, num_prompt_tokens: int) -> int:
        if self.need_truncate and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_request_for_hybrid_prefill(self, request: "Request") -> None:
        params = request.kv_transfer_params
        if (
            params is None
            or not self.need_truncate
            or params.get("_p_side_truncated")
            or getattr(request, "num_prompt_tokens", len(request.prompt_token_ids or [])) <= 1
        ):
            return

        if request.prompt_token_ids is not None:
            request.prompt_token_ids.pop()
        elif request.prompt_embeds is not None:
            request.prompt_embeds = request.prompt_embeds[:-1]
        else:
            return

        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        request.max_tokens = 1
        params["_p_side_truncated"] = True

    def _trim_hybrid_remote_block_ids(self, block_ids: tuple[list[int], ...], prompt_len: int) -> tuple[list[int], ...]:
        if not self.need_truncate or prompt_len <= 1:
            return block_ids

        trimmed_block_ids: list[list[int]] = []
        for group_block_ids, block_size in zip(block_ids, self.block_size):
            if prompt_len % block_size == 1:
                trimmed_block_ids.append(list(group_block_ids[:-1]))
            else:
                trimmed_block_ids.append(list(group_block_ids))
        return tuple(trimmed_block_ids)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeLayerwiseConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert num_computed_tokens % min(self.block_size) == 0
            count = max(self._hybrid_prefill_token_count(len(request.prompt_token_ids)) - num_computed_tokens, 0)
            return count, count > 0

        if params is not None and params.get("do_remote_decode"):
            self._truncate_request_for_hybrid_prefill(request)

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeLayerwiseConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            do_virtual = params.get("do_virtual", False)
            local_block_ids = (blocks.get_block_ids()) if num_external_tokens > 0 else []
            remote_block_ids = self._trim_hybrid_remote_block_ids(local_block_ids, len(request.prompt_token_ids))
            remote_cached_tokens = request.num_computed_tokens
            # Get unhashed blocks to pull from remote.
            logger.debug(
                "MooncakeLayerwiseConnector update_state_after_alloc: add %s to need recv queue", request.request_id
            )
            self._reqs_need_recv[request.request_id] = (
                request,
                [],  # request._all_token_ids,
                local_block_ids,
            )

            params["do_remote_prefill"] = False

            logger.info("Send request: %s to proxy metaserver: %s", request.request_id, params.get("metaserver", None))
            # All parameters here should appear in the returned dict of
            # request_finished in the scheduler side except "request_id".
            # change the format of request_id if vllm-version >= 0.14.0
            external_req_id = get_external_request_id(request.request_id)
            # request_instance is generated exactly once
            # per request lifetime, in this do_remote_prefill branch (it runs
            # once — params["do_remote_prefill"] flips to False below), and
            # reaches P via kv_transfer_params, the only existing D->P channel.
            # Receipt protocol participation is opt-in: the instance is
            # stamped only when a TargetProvider is installed, so a plain-mode
            # engine never advertises receipt identity and P stays on the
            # plain path (its emitter sees an empty instance).
            receipt_enabled = getattr(self, "_target_provider", None) is not None
            request_instance = uuid.uuid4().hex if receipt_enabled else ""
            if receipt_enabled:
                # Also stamp THIS request's own params: they ride vLLM's
                # scheduler->worker serialization, so the D-side worker can
                # build the receipt ledger without cross-process handoff.
                params["request_instance"] = request_instance
            kv_transfer_params = dict(
                token_ids=[],
                request_id=external_req_id,
                request_instance=request_instance,
                do_remote_prefill=False,
                do_remote_decode=True,
                remote_block_ids=remote_block_ids,
                remote_block_size=self.block_size,
                remote_engine_id=self.engine_id,
                remote_host=self.side_channel_host,
                remote_port=self.side_channel_port,
                remote_tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                remote_pcp_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
                remote_dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
                remote_cached_tokens=remote_cached_tokens,
            )
            if not do_virtual:
                # Publish the D-side target BEFORE notifying P: in receipt mode
                # P starts sending on this notification, and receipts arriving
                # before admission would be dropped with no ledger to judge
                # them. Plain mode (no provider) skips publish and notifies.
                notify_p = True
                if receipt_enabled:
                    try:
                        notify_p = bool(self._maybe_publish_target(request, kv_transfer_params, local_block_ids))
                    except Exception as e:
                        logger.error(
                            "Target publish failed for %s (instance=%s): %s", request.request_id, request_instance, e
                        )
                        notify_p = False
                    if not notify_p:
                        # Acquire/validate failures leave no state behind.
                        # Suppressing the P notification means no receipt can
                        # arrive un-owned; the request fails bounded (client
                        # timeout or the receipt watchdog).
                        logger.error(
                            "Receipt request %s: target publish failed; P notification suppressed", request.request_id
                        )
                if notify_p:
                    future = self.executor.submit(
                        self._access_metaserver, url=params.get("metaserver", None), message=kv_transfer_params
                    )

                    def handle_exception(future):
                        if future.exception():
                            logger.error("Access metaserver fail. error=%s. ", future.exception())

                    future.add_done_callback(handle_exception)

        # Layerwise prefiller add request need send
        if params is not None and params.get("do_remote_decode"):
            local_block_ids = list(blocks.get_block_ids())
            logger.debug(
                "MooncakeLayerwiseConnector update_state_after_alloc: add %s to need send queue", request.request_id
            )
            remote_cache_tokens = params["remote_cached_tokens"]
            local_transferred_tokens = remote_cache_tokens
            local_computed_tokens = 0
            self._reqs_need_send_layerwise[request.request_id] = SendReqInfo(
                local_block_ids=local_block_ids,
                local_transferred_tokens=local_transferred_tokens,
                local_computed_tokens=local_computed_tokens,
                request=request,
            )

    def attach_target_provider(self, provider) -> None:
        """Scheduler role: store the provider and the per-request
        target state published via _maybe_publish_target."""
        assert_provider_capabilities(provider)
        self._target_provider = provider
        self._request_leases: dict[str, Any] = {}
        self._target_release_acks: dict[RequestKey, ReleaseAck] = {}
        self._ready_targets: set[RequestKey] = set()
        # Scheduler-owned retention authority, bound via
        # bind_scheduler_retention() by the deployment's scheduler wrapper.
        # Receipt-mode admission fails closed until it is bound: a non-empty
        # retention token must come from a registry that verifies the live
        # scheduler allocation, never from a fabricated string.
        self._retention_registry = None

    def bind_scheduler_retention(self, kv_cache_manager, *, allocation_lock=None) -> None:
        """Bind the scheduler's real KVCacheManager as the retention authority.

        The scheduler wrapper calls this once at init, before any receipt
        request is admitted. Receipt-mode admission fails closed until the
        authority is bound: a non-empty retention token must come from a
        registry that verifies the live scheduler allocation.
        """
        from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import (
            SchedulerRetentionRegistry,
        )

        if self._retention_registry is not None and self._retention_registry._records:
            raise ContractError("Cannot replace an active scheduler retention authority")
        self._retention_registry = SchedulerRetentionRegistry(kv_cache_manager, allocation_lock=allocation_lock)

    def _maybe_publish_target(self, request: "Request", kv_transfer_params: dict, local_block_ids) -> None:
        """Acquire the D-side target right after allocation and
        create the request's ReceiptLedger. Plain mode (no provider attached)
        is a no-op that counts as publishable: returns True when the caller
        may notify P, False when receipt-mode participation must be
        suppressed. Contract failures fail this request loudly — never
        half-published."""
        provider = getattr(self, "_target_provider", None)
        if provider is None:
            return True
        request_instance = kv_transfer_params.get("request_instance", "")
        external_req_id = kv_transfer_params.get("request_id", "")
        request_key = RequestKey(request_instance=request_instance, transfer_generation=0)
        # Block retention: the scheduler-owned registry verifies the live
        # allocation and issues the authoritative token (the contract requires
        # a non-empty token and rejects acquisition without one). Admission is
        # fail-closed: receipt mode without a bound authority never publishes.
        registry = self._retention_registry
        if registry is None:
            logger.error(
                "Receipt request %s: scheduler retention authority is not bound; refusing to publish",
                request.request_id,
            )
            return False
        from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import SerializedKVCacheManager

        if not isinstance(registry._manager, SerializedKVCacheManager):
            raise ContractError("Receipt target admission requires ReceiptScheduler's serialized allocation/free gate")
        allocation = BlockAllocation(
            request=request_key,
            allocation_generation=0,
            group_block_ids=tuple(tuple(group) for group in local_block_ids) or ((0,),),
            manager_block_size_tokens=min(self.block_size),
            prompt_tokens=max(len(request.prompt_token_ids), 1),
        )
        allocation = registry.issue(request.request_id, allocation)
        lease = provider.acquire_target(request_key, allocation)
        # Any subsequent validation/admission failure still leaves real owner
        # references. Keep the lease reachable for the retirement gate.
        self._request_leases[request_instance] = lease
        assert_lease_consistent(lease, request_key)
        wire_lease = target_lease_to_wire(lease)
        manifest = provider.get_target_manifest(lease)
        wire_manifest = target_manifest_to_wire(manifest)
        # P/D consistency: the producer must cover every writer rank the lease
        # assigns, Main must be BF16 (an INT8 producer Main cannot splice into
        # the BF16 D Host pool), and v1 requires one region per backing. The
        # producer TP comes from the lease's sender ranks; the block sizes and
        # Main dtype are the D catalog's own, which the lease views mirror.
        sender_ranks = kv_transfer_params.get("remote_tp_size") or self.vllm_config.parallel_config.tensor_parallel_size
        if sender_ranks != len(manifest.targets):
            raise ContractError("Producer and consumer TP sizes must match the rank-explicit manifest")
        assert_pd_consistent(
            lease,
            producer_tp_size=sender_ranks,
            producer_block_sizes=tuple(self.block_size),
            producer_main_dtype="bfloat16",
        )
        # Startup-check leftovers that depend on the lease contents:
        # manager_block % page is enforced by CacheView.__post_init__; writer
        # coverage is proven here before anything is published.
        prompt_len = allocation.prompt_tokens
        for backing, ranges in lease.writer_plan.ranges_by_backing.items():
            covered = 0
            for token_range in sorted(ranges, key=lambda r: r.start):
                if token_range.start > covered:
                    break
                covered = max(covered, token_range.end)
            if covered < prompt_len:
                raise ContractError(
                    f"writer coverage missing for backing {backing!r}: union ends at {covered} < prompt {prompt_len}"
                )
        # MAIN backings for the ledger's union check: regions that MAIN views
        # bind, deduplicated in view order (v1 single-backing wires
        # backing_id=""; the D ledger resolves it to the first of these).
        backings = tuple(
            dict.fromkeys(
                region.backing_id
                for view in lease.views
                if view.component is Component.MAIN
                for region in lease.regions
                if region.region_id == view.region_binding
            )
        )
        # All real owner receive maps/ledgers must be installed before the
        # metaserver can start P, including its zero-compute Store-hit path.
        provider.admit_target(manifest, request.request_id)
        # Carry the target's mooncake transport endpoint and writer plan to the
        # producer so it can open a session without a separate zmq handshake
        # and apply the lease's Main-writer / Indexer-per-rank分工. Per-engine,
        # shared by every region of the lease.
        if lease.regions:
            kv_transfer_params["remote_te_rpc_endpoint"] = lease.regions[0].transport_endpoint
        kv_transfer_params["remote_target_lease"] = wire_lease
        kv_transfer_params["remote_target_manifest"] = wire_manifest
        # Destination capacity says nothing about whether P has committed KV.
        # Only P's actual Store lookup may select a no-forward restore step.
        logger.info(
            "Published target for req %s (instance=%s, backings=%s, expected_senders=%d)",
            external_req_id,
            request_instance,
            backings,
            sender_ranks,
        )
        return True

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeLayerwiseConnectorMetadata()

        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            registry = getattr(self, "_retention_registry", None)
            if registry is not None:
                meta.scheduled_target_allocations = registry.validate_scheduled(scheduler_output.num_scheduled_tokens)
            # Loop through scheduled reqs and convert to ReqMeta.
            for req_id, (req, token_ids, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                # For the case where there are no remote blocks to pull
                # (block_ids is empty), we don't need to schedule
                # an async read on the worker side.
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    token_ids=token_ids,
                )

            # Clear the list once workers start the transfers
            self._reqs_need_recv.clear()
        else:
            cached_reqs = scheduler_output.scheduled_cached_reqs
            new_reqs = scheduler_output.scheduled_new_reqs
            scheduled_spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
            for req_id, new_blocks in zip(cached_reqs.req_ids, cached_reqs.new_block_ids):
                if req_id in self._reqs_need_send_layerwise and new_blocks is not None:
                    self._reqs_need_send_layerwise[req_id].extend_local_block_ids(new_blocks)
            computed_tokens = dict(
                list(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens))
                + [(x.req_id, x.num_computed_tokens) for x in new_reqs]
            )
            for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
                if req_id in self._reqs_need_send_layerwise:
                    send_req_info = self._reqs_need_send_layerwise[req_id]
                    # update local transferred tokens
                    send_req_info.update_transferred_tokens(
                        round_down(send_req_info.local_computed_tokens, min(self.block_size))
                    )
                    # update local computed tokens, not transfer spec decode tokens
                    spec_decode_tokens = (
                        len(scheduled_spec_decode_tokens[req_id]) if (req_id in scheduled_spec_decode_tokens) else 0
                    )
                    send_req_info.update_computed_tokens(
                        computed_tokens.get(req_id, 0) + scheduled_tokens - spec_decode_tokens
                    )

                    def add_transfer_task(req_id, send_req_info: SendReqInfo, chunk_finish=False):
                        (
                            local_block_ids,
                            local_transed_tokens,
                            local_computed_tokens,
                            request,
                        ) = send_req_info.unpack()
                        meta.add_new_req(
                            request_id=req_id,
                            local_block_ids=local_block_ids,
                            kv_transfer_params=request.kv_transfer_params,
                            token_ids=[],
                            chunk_finish=chunk_finish,
                            remote_cache_tokens=request.kv_transfer_params.get("remote_cached_tokens"),
                            prompt_len=len(request.all_token_ids),
                            local_computed_tokens=local_computed_tokens,
                            local_transed_tokens=local_transed_tokens,
                            remote_target=send_req_info.remote_target,
                        )
                        if send_req_info.remote_target is None:
                            request_meta = meta.requests[req_id]
                            # Target identity and geometry are fixed for this
                            # send lifetime; only step progress changes.
                            send_req_info.remote_target = (
                                request_meta.remote_target_lease,
                                request_meta.remote_target_manifest,
                            )
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "MooncakeLayerwiseConnector build_connector_meta: req_id=%r prompt_len=%s "
                                "local_computed_tokens=%r local_transed_tokens=%r remote_cache_tokens=%s "
                                "chunk_finish=%r local_block_ids=%r remote_block_ids=%s",
                                req_id,
                                len(request.all_token_ids),
                                local_computed_tokens,
                                local_transed_tokens,
                                request.kv_transfer_params.get("remote_cached_tokens"),
                                chunk_finish,
                                local_block_ids,
                                request.kv_transfer_params.get("remote_block_ids"),
                            )

                    # whether chunk finish
                    chunk_finish = send_req_info.local_computed_tokens >= len(send_req_info.request.all_token_ids)

                    add_transfer_task(req_id, send_req_info, chunk_finish=chunk_finish)
                    if chunk_finish:
                        self._reqs_need_send_layerwise.pop(req_id)
        return meta

    def _access_metaserver(self, url, message):
        success = False
        retry = 0
        while retry < 3 and success is False:
            retry += 1
            try:
                self.metaserver_client.post(url, json=message)
                success = True
            except Exception as e:
                logger.error("Failed to connect to metaserver. url=%s, retry=%d. ", url, retry)
                if retry == 3:
                    raise e

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        # ReceiptScheduler retires only after its GPU/queued-step fence.
        # This hook runs earlier and must not release a target that a queued
        # worker step has not entered yet.
        return self._has_retained_target(request), None

    def _is_target_ready(self, request) -> bool:
        instance = (request.kv_transfer_params or {}).get("request_instance")
        lease = getattr(self, "_request_leases", {}).get(instance)
        if lease is None:
            return True
        if lease.request in self._ready_targets:
            return True
        try:
            if self._target_provider.delivery_status(lease)["ready"]:
                self._ready_targets.add(lease.request)
                return True
        except Exception as error:
            logger.warning("All-rank target visibility is not established for %s: %s", request.request_id, error)
        return False

    def _has_retained_target(self, request) -> bool:
        instance = (request.kv_transfer_params or {}).get("request_instance")
        return instance in getattr(self, "_request_leases", {})

    def _release_target_if_any(self, request: "Request") -> bool:
        """Release entry: the provider's release_target normal return is its
        promise that references are dropped; the connector adds its own transfer
        quiescence (in-flight ledger empty for this instance) before counting
        the blocks safe to free. Returns True when the caller may free the
        request's blocks now. On release_target failure the lease is retained
        for diagnosis and the caller must defer freeing: the provider raised "unsafe to
        release", so freeing memory it may still reference would corrupt the
        next request that reuses those blocks. The deferred blocks stay
        unreleased until process exit — bounded by the failure count, and
        strictly safer than a dirty write."""
        provider = getattr(self, "_target_provider", None)
        leases = getattr(self, "_request_leases", None)
        if provider is None or not leases:
            return True
        instance = (request.kv_transfer_params or {}).get("request_instance")
        lease = leases.get(instance) if instance else None
        if lease is None:
            return True
        ack = getattr(self, "_target_release_acks", {}).get(lease.request)
        if ack is None:
            status = getattr(provider, "delivery_status", None)
            if not callable(status):
                return False
            try:
                if not status(lease)["remote_quiesced"]:
                    return False
            except Exception as error:
                logger.warning("Remote DMA quiescence is not established for %s: %s", request.request_id, error)
                return False
            ack = ReleaseAck(
                True, lease.allocation.retention_token, "All admitted TP native writers closed and quiescent"
            )
            self._target_release_acks[lease.request] = ack
        if ack is None or not ack.safe_to_release or ack.retention_token != lease.allocation.retention_token:
            # Request completion/cancellation is not remote DMA quiescence.
            # A worker/control-path proof must precede local owner release.
            return False
        try:
            provider.release_target(lease)
        except Exception as e:
            logger.error(
                "release_target failed for %s (instance=%s): %s -- deferring block free",
                request.request_id,
                instance,
                e,
            )
            return False
        # ReleaseAck aggregation (v1 premise): the receipt chain gated decode
        # on per-rank/per-layer terminal SUCCESS, and the owner release above
        # waited for compute/writeback quiescence — finishing the request
        # therefore certifies the safe-release acknowledgement. Retiring the
        # retention record authorizes the scheduler's existing free path to
        # reclaim the blocks.
        registry = getattr(self, "_retention_registry", None)
        if registry is None:
            logger.error("Missing retention authority for %s; deferring block free", request.request_id)
            return False
        try:
            registry.retire(
                lease.allocation,
                ack,
            )
        except Exception as e:
            logger.error(
                "retire failed for %s (instance=%s): %s -- deferring block free",
                request.request_id,
                instance,
                e,
            )
            return False
        # Preserve the lease on every failed retirement. Otherwise a retry
        # sees no lease and incorrectly authorizes the ordinary free path.
        leases.pop(instance, None)
        self._target_release_acks.pop(lease.request, None)
        getattr(self, "_ready_targets", set()).discard(lease.request)
        logger.info("Released target for %s (instance=%s)", request.request_id, instance)
        return True

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        return self._has_retained_target(request), None


class MooncakeLayerwiseConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())

        if TransferEngine is None:
            raise RuntimeError("mooncake is not available")
        logger.info("Initializing Mooncake work %s", engine_id)

        # Metadata.
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        self.kv_cache_specs: list[KVCacheSpec] = [spec.kv_cache_spec for spec in self.kv_cache_config.kv_cache_groups]
        self.local_engine_id: str = " "
        self.engine_id = engine_id
        self.dp_rank: int = vllm_config.parallel_config.data_parallel_rank
        self.tp_rank: int = get_tensor_model_parallel_rank()
        self.tp_size: int = vllm_config.parallel_config.tensor_parallel_size
        self.pcp_size: int = vllm_config.parallel_config.prefill_context_parallel_size
        self.pcp_rank: int = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size: int = vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_rank: int = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0
        self.tp_group = get_tp_group()
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.use_mla = self.vllm_config.model_config.use_mla
        self.request_map = dict[str, str]()
        self.use_attn_mamba_hybrid = False
        if self.use_mla:
            self.total_num_kv_heads = 1
        else:
            self.total_num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + self.dp_rank * self.pcp_size * self.tp_size
            + self.pcp_rank * self.tp_size
        )
        self.handshake_port = self.side_channel_port + self.tp_rank
        self.sockets: dict = {}
        logger.info("Initializing Mooncake work %s", engine_id)
        inject_qos(vllm_config.kv_transfer_config.get_from_extra_config("qos_priority", PD_QOS_DEFAULT))
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_recv_layer_thread: KVCacheRecvingLayerThread | None = None
        self.kv_send_layer_thread: KVCacheSendingLayerThread | None = None
        self.layer_storage_slots: dict[int, tuple[int, ...]] = {}
        self.storage_send_done_events: list[threading.Event] = []
        self._storage_send_errors: dict[int, str] = {}
        self._pending_reuse_layers: set[int] = set()
        self._cache_write_events: list[torch.npu.Event | None] = [None] * self.total_layers
        # Receipt-mode / reuse-ledger state. The slot
        # ledger is created in _init_layerwise_reuse_state once the planner's
        # layer->slot mapping exists.
        self._slot_ledger: SlotLedger | None = None
        self._slot_ledger_active = False
        self._layer_keys: dict[int, tuple[TransferKey, ...]] = {}
        self._compute_done_events: list[torch.npu.Event | None] = [None] * self.total_layers
        self._no_compute_layers: set[int] = set()
        self._target_provider = None
        self._receipt_send_drops = 0  # receipt messages lost after retries (watchdog backstop)
        # Source consumer v1 adapter; installed on the producer so the Store
        # registry can drive source-lifecycle callbacks. It delegates to this
        # worker's slot ledger and send path, never starting a second send.
        self._reuse_source_consumer: Any = None
        self._manifest_source_writer = None
        self._source_request_metadata = {}
        self._source_contract_bound = False
        if self.vllm_config.kv_transfer_config.is_kv_producer:
            from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_source_consumer import (
                LayerwiseSourceConsumer,
            )

            self._reuse_source_consumer = LayerwiseSourceConsumer(self)

        self.block_size: list[int] = [spec.block_size for spec in self.kv_cache_specs]
        self.kernel_block_size_scale: list[int] = [1 for _ in range(self.num_kv_cache_groups)]
        self.layer_metadata: dict[str, LayerMetadata] = {}
        self.sparse_sfa_c8_layers: set[str] = set()
        self.attn_resharding_group_idx = set[int]()

        self.enable_kv_quant = (
            vllm_config.quant_config.enable_fa_quant if vllm_config.quant_config is not None else False
        )
        self.enable_c8_quant = (
            vllm_config.quant_config.enable_c8_quant if vllm_config.quant_config is not None else False
        )
        self.pd_head_ratio = get_ascend_config().pd_head_ratio
        self.num_head_replica = get_ascend_config().num_head_replica
        self.resharding_stream = None
        if self.pd_head_ratio > 1 or self.enable_kv_quant or self.enable_c8_quant:
            self.resharding_stream = torch.npu.Stream()

        self.remote_poller = zmq.Poller()  # type: ignore
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.encoder = msgspec.msgpack.Encoder()

        self.index_to_name = defaultdict(list)
        self.remote_layer_metadata: dict[str, dict[int, dict[str, LayerMetadata]]] = SizedDict()
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds
        self.k_buffer: torch.Tensor | None = None
        self.v_buffer: torch.Tensor | None = None
        self.virtual_request: set[str] = set()
        self._invalid_block_ids: set[int] = set()
        self._failed_receives: set[str] = set()
        self._recving_metadata: dict[str, ReqMeta] = {}

    def _init_layerwise_reuse_state(self) -> None:
        slot_by_addr: dict[int, int] = {}
        layer_storage_slots: dict[int, tuple[int, ...]] = {}
        for layer_idx, layer_names in self.index_to_name.items():
            slots: list[int] = []
            for layer_name in layer_names:
                for addr in self.layer_metadata[layer_name].kv_caches_base_addr:
                    slot_id = slot_by_addr.setdefault(addr, len(slot_by_addr))
                    if slot_id not in slots:
                        slots.append(slot_id)
            layer_storage_slots[layer_idx] = tuple(slots)
        self.layer_storage_slots = layer_storage_slots
        self.storage_send_done_events = []
        for _ in range(len(slot_by_addr)):
            event = threading.Event()
            event.set()
            self.storage_send_done_events.append(event)
        # the ledger is keyed by the same address-derived slot ids,
        # sourced from the planner's mapping above — never re-derived by modulo.
        self._slot_ledger = SlotLedger(lambda layer_idx: self.layer_storage_slots.get(layer_idx, ()))

    def _mark_layer_reuse_pending(self, layer_idx: int) -> None:
        if layer_idx not in self.layer_storage_slots:
            raise RuntimeError(f"Mooncake layerwise reuse mapping is missing layer {layer_idx}")
        if layer_idx in self._pending_reuse_layers:
            return
        self._pending_reuse_layers.add(layer_idx)
        for slot_id in self.layer_storage_slots[layer_idx]:
            self._storage_send_errors.pop(slot_id, None)
            self.storage_send_done_events[slot_id].clear()

    def _complete_layer_reuse(self, layer_idx: int, error: str | None) -> None:
        self._complete_layer_tickets(layer_idx, error)
        if layer_idx not in self._pending_reuse_layers:
            return
        self._pending_reuse_layers.remove(layer_idx)
        for slot_id in self.layer_storage_slots[layer_idx]:
            if error is not None:
                self._storage_send_errors[slot_id] = error
            self.storage_send_done_events[slot_id].set()

    def wait_for_layer_reuse(self, layer_idx: int) -> None:
        send_thread = self.kv_send_layer_thread
        if send_thread is None:
            return
        if layer_idx not in self.layer_storage_slots:
            raise RuntimeError(f"Mooncake layerwise reuse mapping is missing layer {layer_idx}")
        if self._slot_ledger is not None and self._slot_ledger_active:
            # Ledger path: resolve the layer to its
            # slots, snapshot the current generation per slot at entry, then
            # apply the three-branch rule. SlotError (failed generation or
            # protocol misuse) surfaces as RuntimeError; TimeoutError likewise.
            try:
                self._slot_ledger.wait(layer_idx, LAYERWISE_SLOT_WAIT_TIMEOUT_SECONDS)
                return
            except SlotError as slot_error:
                raise RuntimeError(f"Mooncake layerwise slot reuse failed: {slot_error}") from slot_error
            except TimeoutError as timeout_error:
                raise RuntimeError(f"Mooncake layerwise slot reuse timed out: {timeout_error}") from timeout_error
        for slot_id in self.layer_storage_slots[layer_idx]:
            event = self.storage_send_done_events[slot_id]
            while not event.wait(timeout=LAYERWISE_REUSE_WAIT_LOG_INTERVAL_SECONDS):
                if not send_thread.is_alive():
                    raise RuntimeError("Mooncake send thread stopped while waiting for layerwise buffer reuse")
                logger.info("Waiting for Mooncake transfer to release physical KV slot %d", slot_id)
            error = self._storage_send_errors.get(slot_id)
            if error is not None:
                raise RuntimeError(f"Mooncake failed to transfer physical KV slot {slot_id}: {error}")

    def attach_target_provider(self, provider) -> None:
        """Single capability-probe entry, called by
        AscendMultiConnector. Raises on any capability gap -> the
        engine never starts in a degraded mode."""
        assert_provider_capabilities(provider)
        self._target_provider = provider
        if self.kv_recv_layer_thread is not None:
            self.kv_recv_layer_thread.receipt_provider = provider

    def send_layer_result(self, req_meta: ReqMeta, msg: tuple) -> None:
        """Best-effort receipt transport: REQ/ACK with 3 retries,
        copied from send_done_send_signal. Version mismatch fails before
        sending. Failures are logged, not raised: the D-side watchdog bounds
        the stall instead."""
        if msg[1] != CONTRACT_VERSION:
            raise RuntimeError(f"receipt protocol version mismatch: {msg[1]} != {CONTRACT_VERSION}")
        if req_meta.remote_host is None or req_meta.remote_port is None:
            logger.debug("Receipt message dropped: no remote endpoint for request yet")
            return
        path = make_zmq_path("tcp", req_meta.remote_host, req_meta.remote_port)
        encoded = self.encoder.encode(msg)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore
                    sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
                    ensure_zmq_send(sock, encoded, f"{req_meta.remote_host}:{req_meta.remote_port}")
                    if not sock.poll(int(self.timeout * 1000), zmq.POLLIN):  # type: ignore[attr-defined]
                        raise TimeoutError(
                            f"Timed out waiting for ACK from {req_meta.remote_host}:{req_meta.remote_port}"
                        )
                    ack = sock.recv()
                    if ack != b"ACK":
                        raise ValueError(f"Unexpected ACK response: {ack}")
                    return
            except Exception as e:
                if attempt >= max_retries:
                    self._receipt_send_drops += 1
                    logger.error(
                        "DROPPED receipt message type=%s for instance=%s after %d attempts "
                        "(total drops=%d; D-side watchdog bounds the stall). error=%s",
                        msg[0],
                        msg[2],
                        max_retries,
                        self._receipt_send_drops,
                        e,
                    )

    def _build_transfer_key(self, req_id: str, req_meta: ReqMeta, layer_idx: int) -> TransferKey:
        """Slot-ledger ticket identity for one request's read of this layer.
        On the P side the ledger tracks source readers only; the D-facing
        component/backing detail lives in the wire messages instead."""
        end_token = min(req_meta.local_computed_tokens or req_meta.prompt_len, req_meta.prompt_len)
        start_token = min(req_meta.local_transed_tokens, end_token)
        return TransferKey(
            request=RequestKey(request_instance=req_meta.request_instance, transfer_generation=0),
            rank=self.tp_rank,
            logical_layer=layer_idx,
            operation_id=layer_idx,
            component=Component.MAIN,
            token_range=TokenRange(start=start_token, end=end_token),
            target_scope=TargetScope(backing_id="", backing_generation=0, consumer_rank=None),
        )

    def _complete_layer_tickets(self, layer_idx: int, error: str | None) -> None:
        """Receipt mode: finish the layer's registered slot
        tickets. Success marks done+quiesced (source readers released); failure
        keeps the source held (quiesced=False) — the abort path
        owns that request's cleanup."""
        keys = self._layer_keys.pop(layer_idx, None)
        ledger = self._slot_ledger
        if ledger is None or not keys:
            return
        status = Status.FAILED if error else Status.SUCCESS
        quiesced = error is None
        for key in keys:
            ledger.complete(layer_idx, key, status, source_quiesced=quiesced)

    def _run_startup_validations(self) -> None:
        """Startup fail-fast checks. Any failure aborts startup —
        never degrade silently."""
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        if pp_size != 1:
            raise RuntimeError(f"Mooncake layerwise receipt mode requires PP=1, got PP={pp_size}")
        if not self.index_to_name:
            raise RuntimeError("Mooncake layerwise receipt mode found no physical layers to manage")
        for layer_idx, layer_names in self.index_to_name.items():
            if not layer_names:
                raise RuntimeError(f"Mooncake layerwise mapping for layer {layer_idx} is empty")
            for layer_name in layer_names:
                if layer_name not in self.layer_metadata:
                    raise RuntimeError(f"Mooncake layerwise metadata missing for layer {layer_name!r}")
        # manager_block % page and writer coverage are enforced per request at
        # publish time (CacheView.__post_init__ + writer-plan check in
        # _maybe_publish_target) because they depend on the provider's lease contents.

    def create_kv_buffer(self, first_kv_cache_tuple):
        alignment = 2 * 1024 * 1024
        first_k_cache = first_kv_cache_tuple[0]
        first_v_cache = first_kv_cache_tuple[1]
        if self.pd_head_ratio > 1 or self.enable_kv_quant or self.enable_c8_quant:
            # regesit kv buffer
            if self.enable_kv_quant:
                k_dtype = torch.int8
                v_dtype = first_v_cache.dtype
            elif self.enable_c8_quant:
                k_dtype = torch.int8
                v_dtype = torch.int8
            else:
                k_dtype = first_k_cache.dtype
                v_dtype = first_v_cache.dtype
            self.k_buffer = torch.zeros(first_k_cache.numel() + alignment, dtype=k_dtype, device=first_k_cache.device)
            self.k_buffer = align_memory(self.k_buffer, alignment)[: first_k_cache.numel()].view(
                -1, first_k_cache.shape[-1]
            )
            self.v_buffer = torch.zeros(first_v_cache.numel() + alignment, dtype=v_dtype, device=first_k_cache.device)
            self.v_buffer = align_memory(self.v_buffer, alignment)[: first_v_cache.numel()].view(
                -1, first_v_cache.shape[-1]
            )
        for tensor in (self.k_buffer, self.v_buffer):
            assert tensor.data_ptr() % alignment == 0, "The address of the registered kv cache should be aligned to 2M"
            ret_value = self.engine.register_memory(tensor.data_ptr(), tensor.numel() * tensor.element_size())
            logger.info("Register memory buffer for transfer, buffer size:%s", tensor.numel() * tensor.element_size())
            if ret_value != 0:
                raise RuntimeError("Mooncake memory registration failed. ")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""
        target_manager = getattr(self, "target_owner_manager", None)
        if target_manager is not None:
            kv_caches = target_manager.get_target_transfer_caches(kv_caches)
        layer2group_ids: dict[str, int] = {}
        kv_cache_groups = self.kv_cache_config.kv_cache_groups
        for i, kv_cache_group_spec in enumerate(kv_cache_groups):
            for layer_name in kv_cache_group_spec.layer_names:
                layer2group_ids[layer_name] = i

        use_mamba, use_attn = False, False
        conv_total_padding_size = 0
        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            for layer_name in get_kv_cache_tensor_layers(kv_cache_tensor):
                layer_kv_cache_spec = self.kv_cache_specs[layer2group_ids[layer_name]]
                if isinstance(layer_kv_cache_spec, MambaSpec):
                    use_mamba = True
                    conv_shape, conv_dtype = layer_kv_cache_spec.shapes[0], layer_kv_cache_spec.dtypes[0]
                    conv_total_padding_size = (
                        self.kv_cache_config.num_blocks * math.prod(conv_shape) * get_dtype_size(conv_dtype)
                    )
                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    use_attn = True
            if use_mamba and use_attn:
                self.use_attn_mamba_hybrid = True
                break

        ptrs = []
        lengths = []
        kv_buffer = None
        group_block_size_scales: dict[int, int] = {}
        for layer_name, kv_cache_tuple in kv_caches.items():
            if isinstance(kv_cache_tuple, (list, tuple)) is False:
                kv_cache_tuple = [kv_cache_tuple]
            layer_kv_group_id = layer2group_ids[layer_name]
            layer_kv_cache_spec = kv_cache_groups[layer_kv_group_id].kv_cache_spec
            if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
            # Read the layout flags directly: some Ascend versions expose the
            # cache spec flags without the corresponding utils helpers.
            is_mla = isinstance(layer_kv_cache_spec, MLAAttentionSpec)
            sparse_sfa_c8 = is_mla and bool(getattr(layer_kv_cache_spec, "cache_sparse_sfa_c8", False))
            sparse_sfa_turboquant = is_mla and bool(getattr(layer_kv_cache_spec, "cache_sparse_sfa_turboquant", False))
            if sparse_sfa_turboquant:
                raise NotImplementedError(
                    "Mooncake layerwise connector does not support packed SFA main KV caches other than C8."
                )
            if sparse_sfa_c8:
                # Attention already packs quantized NoPE, BF16 RoPE and scales
                # into one byte tensor on both P and D. Transfer it unchanged.
                if len(kv_cache_tuple) != 1 or not kv_cache_tuple[0].is_contiguous():
                    raise NotImplementedError("SFA C8 layerwise transfer requires one contiguous packed cache tensor.")
                self.sparse_sfa_c8_layers.add(layer_name)
            if (
                not sparse_sfa_c8
                and "indexer" not in layer_name.lower()
                and (
                    (
                        self.pd_head_ratio > 1
                        and (isinstance(layer_kv_cache_spec, (FullAttentionSpec, SlidingWindowSpec)))
                    )
                    or self.enable_kv_quant
                    or self.enable_c8_quant
                )
            ):
                self.attn_resharding_group_idx.add(layer_kv_group_id)
                if kv_buffer is None:
                    kv_buffer = kv_cache_tuple
            single_layer_meta = LayerMetadata([], [], [], [])
            for single_kv_cache in kv_cache_tuple:
                block_start_rank = 1
                num_blocks = self.kv_cache_config.num_blocks
                tensor_num_blocks = single_kv_cache.shape[0]
                assert tensor_num_blocks % num_blocks == 0, (
                    "The external block size must be an integer multiple of the kernel block size."
                )
                block_size_scale = tensor_num_blocks // num_blocks
                group_block_size_scale = group_block_size_scales.setdefault(
                    layer_kv_group_id,
                    block_size_scale,
                )
                if group_block_size_scale != block_size_scale and target_manager is None:
                    raise NotImplementedError(
                        "Mooncake layerwise connector does not support KV cache components "
                        "with different block size scales in one group, including DCP replicated indexer cache."
                    )
                block_shape = single_kv_cache.shape[block_start_rank:]
                single_layer_meta.tensor_group_idx.append(layer_kv_group_id)
                single_layer_meta.kv_caches_base_addr.append(single_kv_cache.data_ptr())
                single_layer_meta.block_len.append(single_kv_cache.element_size() * math.prod(block_shape))
                single_layer_meta.block_size_scale.append(block_size_scale)
                self.kernel_block_size_scale[layer2group_ids[layer_name]] = block_size_scale
                if single_kv_cache.data_ptr() not in ptrs and not self.use_attn_mamba_hybrid:
                    ptrs.append(single_kv_cache.data_ptr())
                    lengths.append(
                        num_blocks * single_kv_cache.element_size() * math.prod(block_shape) * block_size_scale
                    )
                logger.info("layer: %s, num_blocks: %s, block_shape: %s", layer_name, num_blocks, block_shape)
            self.layer_metadata[layer_name] = single_layer_meta

        if self.use_attn_mamba_hybrid:
            for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
                tensor_addrs = []
                for layer_name in get_kv_cache_tensor_layers(kv_cache_tensor):
                    tensor_addrs.extend(self.layer_metadata[layer_name].kv_caches_base_addr)
                    if "mtp" in layer_name:
                        tensor_addrs.append(min(tensor_addrs) - conv_total_padding_size)
                assert min(set(tensor_addrs)) % (2 * 1024 * 1024) == 0, "Tensor start addr is not align with 2M."
                ptrs.append(min(set(tensor_addrs)))
                lengths.append(kv_cache_tensor.size)

        register_locations = None
        if target_manager is not None:
            components, _, _ = target_manager.get_target_cache_identities()
            pool = target_manager.get_mooncake_host_pool()
            atoms = []
            for name, caches in kv_caches.items():
                for tensor in caches if isinstance(caches, (tuple, list)) else (caches,):
                    start = tensor.data_ptr()
                    span = tensor.element_size() + sum(
                        (size - 1) * stride * tensor.element_size()
                        for size, stride in zip(tensor.shape, tensor.stride())
                    )
                    if components[name] is Component.MAIN:
                        if not pool.data_ptr <= start < start + span <= pool.data_ptr + pool.nbytes:
                            raise ContractError("Host target registration extends outside the existing pool")
                        allocation, location = ("host", pool.data_ptr), f"npu:{pool.topology.device_id}"
                    else:
                        storage = tensor.untyped_storage()
                        if not storage.data_ptr() <= start < start + span <= storage.data_ptr() + storage.nbytes():
                            raise ContractError("Device target registration extends outside its storage")
                        allocation, location = ("device", storage.data_ptr()), "*"
                    atoms.append(DsaRegisterAtom(start, start + span, location, allocation))
            # Reuse the native-bounded DSA packer, keeping complete normal
            # components in one MR while permitting large shared backings.
            bounded = collect_bounded_register_regions(atoms)
            register_regions = RegisterRegions(ptrs=bounded.ptrs, lengths=bounded.lengths)
            register_locations = bounded.locations
        elif self.use_attn_mamba_hybrid:
            register_regions = RegisterRegions(ptrs=ptrs, lengths=lengths)
        else:
            # For normal attention / sparse-c8 KV cache, register merged memory
            # ranges while keeping layer metadata at logical tensor addresses.
            register_regions = collect_storage_merged_register_regions(kv_caches)

        validate_register_region_count(register_regions)
        if register_locations is None:
            global_te.register_buffer(register_regions.ptrs, register_regions.lengths)
        else:
            global_te.register_buffer(register_regions.ptrs, register_regions.lengths, register_locations)

        if kv_buffer is not None:
            self.create_kv_buffer(kv_buffer)

        model_type = self.vllm_config.model_config.hf_text_config.model_type
        num_attn_module = 2 if model_type in ("longcat_flash", "longcat_flash_ngram") else 1
        mtp_layer_name = ""
        for layer_name in kv_caches:
            if "mtp" in layer_name:
                mtp_layer_name = layer_name
                continue
            layer_idx = extract_layer_index(layer_name, num_attn_module)
            self.index_to_name[layer_idx].append(layer_name)
        for layer_names in self.index_to_name.values():
            layer_names.sort(key=lambda name: "indexer" in name.lower())
            primary_group_idx = self.layer_metadata[layer_names[0]].tensor_group_idx[0]
            if target_manager is None:
                assert all(
                    self.layer_metadata[name].tensor_group_idx[0] == primary_group_idx for name in layer_names
                ), f"Mooncake layerwise merged components must share one KV cache group. layer_names={layer_names}"
        if mtp_layer_name != "":
            self.index_to_name[max(self.index_to_name.keys()) + 1].append(mtp_layer_name)
        if self.index_to_name:
            self.total_layers = max(self.total_layers, max(self.index_to_name) + 1)

        self._init_layerwise_reuse_state()
        self._run_startup_validations()

        # After KV Caches registered, start the sending or receiving thread.
        metadata = MooncakeAgentMetadata(
            te_rpc_port=self.te_rpc_port,
            layer_metadata=self.layer_metadata,
        )
        if self.vllm_config.kv_transfer_config.is_kv_producer:
            ready_event = threading.Event()
            self.kv_send_layer_thread = KVCacheSendingLayerThread(
                engine=self.engine,
                vllm_config=self.vllm_config,
                kv_cache_config=self.kv_cache_config,
                kv_cache_specs=self.kv_cache_specs,
                attn_resharding_group_idx=self.attn_resharding_group_idx,
                total_layers=self.total_layers,
                ready_event=ready_event,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                pd_head_ratio=self.pd_head_ratio,
                num_head_replica=self.num_head_replica,
                layer_metadata=self.layer_metadata,
                use_mla=self.use_mla,
                use_attn_mamba_hybrid=self.use_attn_mamba_hybrid,
                k_buffer=self.k_buffer,
                v_buffer=self.v_buffer,
                enable_kv_quant=self.enable_kv_quant,
                enable_c8_quant=self.enable_c8_quant,
                resharding_stream=self.resharding_stream,
                callback_func=self.send_done_send_signal,
                reuse_completion_callback=self._complete_layer_reuse,
            )
            self.kv_send_layer_thread.start()
            self.kv_send_layer_thread.layer_result_sender = self.send_layer_result
            ready_event.wait()
            self._manifest_source_writer = ManifestSourceWriter(
                self.tp_rank,
                self.kv_send_layer_thread.send_queue,
                resolve_manifest_addresses,
                ManifestPeerProofs(self.tp_rank),
            )

        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            ready_event = threading.Event()
            self.kv_recv_layer_thread = KVCacheRecvingLayerThread(
                self.tp_rank,
                self.side_channel_port,
                self.tp_size,
                self.pd_head_ratio,
                self.engine_id,
                metadata,
                ready_event,
            )
            self.kv_recv_layer_thread.start()
            ready_event.wait()

    def get_finished(self, finished_req_ids=None) -> tuple[set[str], set[str]]:
        delivery = getattr(self, "_target_delivery", None)
        if delivery is not None and finished_req_ids:
            delivery.abort_requests(finished_req_ids)
        writer = getattr(self, "_manifest_source_writer", None)
        if writer is not None and finished_req_ids:
            writer.finish_requests(finished_req_ids)
        done_recving = (
            self.kv_recv_layer_thread.get_and_clear_done_requests(  # type: ignore[union-attr]
            )
            if self.vllm_config.kv_transfer_config.is_kv_consumer
            else set()
        )
        done_recving = {self.request_map[s] for s in done_recving if s in self.request_map}
        done_recving.update(self.virtual_request)
        self.virtual_request = set()

        failed_recving = (
            self.kv_recv_layer_thread.get_and_clear_failed_requests()
            if self.vllm_config.kv_transfer_config.is_kv_consumer and self.kv_recv_layer_thread is not None
            else set()
        )
        failed_recving = {self.request_map[s] for s in failed_recving if s in self.request_map}
        for req_id in failed_recving:
            if req_id in getattr(self, "_admitted_target_blocks", {}):
                # Main/Indexer pool IDs may overlap. The deployed vLLM
                # transfer-results API carries unambiguous request failures.
                self._failed_receives.add(req_id)
                continue
            if meta := self._recving_metadata.get(req_id):
                self._invalid_block_ids.update(block_id for group in meta.local_block_ids for block_id in group)
        for req_id in done_recving.union(failed_recving):
            self._recving_metadata.pop(req_id, None)
            exact_target = req_id in getattr(self, "_admitted_target_blocks", {})
            if not exact_target:
                self.request_map.pop(get_external_request_id(req_id), None)
            if self.kv_recv_layer_thread is not None:
                # Exact targets retain their internal identity and block map
                # until owner retirement: late failure/quiescence proofs must
                # not be lost or attributed to another EngineCore lifetime.
                self.kv_recv_layer_thread.forget_request(
                    req_id if exact_target else get_external_request_id(req_id), failed=req_id in failed_recving
                )
        if len(done_recving) > 0:
            logger.info(
                "Number of completed KV cache recv requests: %s, receive requests: %s", len(done_recving), done_recving
            )
        return set(), done_recving

    def take_failed_receives(self):
        result = self._failed_receives
        self._failed_receives = set()
        return result

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Return and clear the set of block IDs that failed to load.
        This is called by the scheduler to identify blocks that need
        to be retried after a transfer failure.
        """
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    # {(ip, port)]: {local_block_ids: [], remote_block_ids: {}}}
    def _get_kv_split_metadata(self, req_meta: ReqMeta, req_idx: int, req_id: str, group_idx: int):
        remote_pcp_size = req_meta.remote_pcp_size
        remote_dcp_size = req_meta.remote_dcp_size
        remote_tp_size = req_meta.remote_tp_size
        remote_hosts = [req_meta.remote_host]
        remote_port = req_meta.remote_port
        local_transed_tokens = max(req_meta.remote_cache_tokens, req_meta.local_transed_tokens)
        # local_transed_tokens tokens that have already been transmitted on the local side
        local_computed_tokens = req_meta.local_computed_tokens
        prompt_len = req_meta.prompt_len
        p_parallel_info = parallel_info(
            tp_size=self.tp_size,
            pcp_size=self.pcp_size,
            dcp_size=self.dcp_size,
            pd_head_ratio=self.pd_head_ratio,
            use_mla=self.use_mla,
        )
        d_parallel_info = parallel_info(
            tp_size=remote_tp_size,
            pcp_size=remote_pcp_size,
            dcp_size=remote_dcp_size,
            pd_head_ratio=self.pd_head_ratio,
            use_mla=self.use_mla,
        )
        cp_size = self.pcp_size * self.dcp_size
        # to_trans_idx all tokens that have been processed up to the current step
        if req_meta.chunk_finish:
            to_trans_idx = math.ceil(local_computed_tokens / self.block_size[group_idx])
        else:
            to_trans_idx = math.floor(local_computed_tokens / self.block_size[group_idx])
        prompt_block_size = math.ceil(prompt_len / self.block_size[group_idx])
        #
        num_local_blocks = prompt_block_size // cp_size + int(
            (prompt_block_size % cp_size) > (self.pcp_rank * self.dcp_size + self.dcp_rank)
        )
        already_send_blocks = to_trans_idx // cp_size + int(
            (to_trans_idx % cp_size) > (self.pcp_rank * self.dcp_size + self.dcp_rank)
        )
        if num_local_blocks == already_send_blocks:
            req_meta.chunk_finish = True
        transed_idx = math.floor(local_transed_tokens / self.block_size[group_idx])

        p_cp_group = get_cp_group(self.tp_size, self.total_num_kv_heads, self.dcp_size)
        d_cp_group = get_cp_group(remote_tp_size, self.total_num_kv_heads, remote_dcp_size)
        logger.debug("Compute cp group for P&D req_id=%r p_cp_group=%r d_cp_group=%r", req_id, p_cp_group, d_cp_group)

        cp_ratio = len(p_cp_group) // len(d_cp_group)
        if cp_ratio == 0:
            selected_p_cp_groups = p_cp_group
            selected_d_cp_groups = d_cp_group
        else:
            x = req_idx % cp_ratio
            start = x * len(d_cp_group)
            selected_p_cp_groups = p_cp_group[start : (start + len(d_cp_group))]
            selected_d_cp_groups = d_cp_group
        assert len(selected_p_cp_groups) == len(selected_d_cp_groups)

        p_head_group_rank = (self.tp_rank - self.dcp_rank) // self.dcp_size
        selected_p_cp_group = []
        selected_d_cp_group = []
        for idx, cp_group in enumerate(selected_p_cp_groups):
            if p_head_group_rank in cp_group:  # Check whether the rank is in selected_p_cp_groups
                selected_p_cp_group = cp_group
                selected_d_cp_group = selected_d_cp_groups[idx]
        if len(selected_p_cp_group) == 0:
            return {}

        logger.debug(
            "MooncakeLayerwiseConnector _get_kv_split_metadata req_id=%r "
            "P-side selected head_group cp group: %s, D-side selected head_group cp group: %s",
            req_id,
            selected_p_cp_group,
            selected_d_cp_group,
        )

        context_parallel_parameters_check(
            remote_pcp_size, remote_dcp_size, p_parallel_info, d_parallel_info, self.total_num_kv_heads
        )
        p_rank_block_mapping, d_block_rank_mapping, pd_head_mapping, d_trans_count_mapping = (
            get_local_remote_block_port_mappings(
                to_trans_idx,
                p_parallel_info,
                d_parallel_info,
                remote_hosts,
                remote_port,
                selected_p_cp_group,
                selected_d_cp_group,
                prompt_len,
                self.block_size[group_idx],
                req_meta,
                self.total_num_kv_heads,
                req_id,
            )
        )
        transfer_mappings = get_transfer_mappings(
            p_rank_block_mapping,
            d_block_rank_mapping,
            pd_head_mapping,
            d_trans_count_mapping,
            req_meta,
            group_idx,
            p_parallel_info,
            req_id,
            transed_idx,
            to_trans_idx,
            self.tp_rank,
            self.pcp_rank,
            self.dcp_rank,
        )
        return transfer_mappings

    def _get_kv_split_metadata_for_mamba(self, req_meta: ReqMeta, req_idx: int, req_id: str, group_idx: int):
        assert self.tp_size >= req_meta.remote_tp_size, (
            "Mamba group prefill TP_size must equal or larger than decode TP_size."
        )
        remote_tp_size = req_meta.remote_tp_size
        tp_raito = self.tp_size // remote_tp_size
        remote_host = req_meta.remote_host
        remote_port = req_meta.remote_port + self.tp_rank // tp_raito
        transfer_mappings: dict[tuple[str, int], dict[str, Any]] = {}
        if req_meta.chunk_finish:
            transfer_mappings[(remote_host, remote_port)] = {
                "local_block_ids": req_meta.local_block_ids[group_idx],
                "remote_block_ids": req_meta.remote_block_ids[group_idx],
                "trans_count": self.tp_size // remote_tp_size,
            }

        return transfer_mappings

    def _align_remote_block_ids(self, req_meta: ReqMeta):
        remote_block_size = req_meta.remote_block_size
        remote_block_ids = req_meta.remote_block_ids
        for i in range(self.num_kv_cache_groups):
            if isinstance(self.kv_cache_specs[i], MambaSpec):
                continue
            if remote_block_size[i] != self.block_size[i] and len(req_meta.remote_block_ids[i]) > 0:
                assert remote_block_size[i] > self.block_size[i] and remote_block_size[i] % self.block_size[i] == 0, (
                    "Remote block size must be divisible by local block size."
                )
                assert self.pcp_size * self.dcp_size * req_meta.remote_pcp_size * req_meta.remote_dcp_size == 1, (
                    "Context parallel does not support different P/D block size now."
                )
                pd_block_size_ratio = remote_block_size[i] // self.block_size[i]
                remtote_block_ids_with_scale = [
                    block_id * pd_block_size_ratio + j
                    for block_id in remote_block_ids[i]
                    for j in range(pd_block_size_ratio)
                ]
                req_meta.remote_block_ids[i] = remtote_block_ids_with_scale

    def _get_kernel_block_ids(self, block_ids):
        for i in range(self.num_kv_cache_groups):
            if isinstance(self.kv_cache_specs[i], MambaSpec):
                continue
            if len(block_ids[i]) > 0:
                block_ids[i] = [
                    block_id * self.kernel_block_size_scale[i] + j
                    for block_id in block_ids[i]
                    for j in range(self.kernel_block_size_scale[i])
                ]
        return block_ids

    def start_load_kv(self, metadata: MooncakeLayerwiseConnectorMetadata):
        """Track D-side receives without resetting forward/MTP send state."""
        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            for req_id, meta in metadata.requests.items():
                if meta.do_virtual:
                    self.virtual_request.add(req_id)
                    continue
                external_req_id = get_external_request_id(req_id)
                assert self.kv_recv_layer_thread is not None
                mapping_key = req_id if req_id in getattr(self, "_admitted_target_blocks", {}) else external_req_id
                self.request_map[mapping_key] = req_id
                self._recving_metadata[req_id] = meta
            specs = getattr(metadata, "receipt_specs", None)
            if specs:
                self._install_receipt_specs(specs)

    def _install_admitted_target(self, req_id, lease, ledger) -> None:
        """Synchronous worker-ready acknowledgement, before any P notification."""
        thread = self.kv_recv_layer_thread
        if thread is None:
            raise ContractError("Target receive thread must exist before manifest admission")
        instance = lease.request.request_instance
        external_req_id = get_external_request_id(req_id)
        with thread.lock:
            if instance in thread.ledger_by_req or req_id in self.request_map:
                raise ContractError("Target receive request identity is already installed")
            if not hasattr(self, "_admitted_target_blocks"):
                self._admitted_target_blocks = {}
            ledger.lease = lease
            self.request_map[req_id] = req_id
            self._admitted_target_blocks[req_id] = lease.allocation.group_block_ids
            thread.ledger_by_req[instance] = ledger
            thread.instance_to_req[instance] = req_id
            thread.receipt_mode_reqs.add(external_req_id)
            thread.receipt_deadlines[instance] = time.monotonic() + RECEIPT_WATCHDOG_SECONDS

    def _retire_admitted_target(self, req_id, lease, ledger):
        thread = self.kv_recv_layer_thread
        instance = lease.request.request_instance
        external = get_external_request_id(req_id)
        with thread.lock:
            if thread.ledger_by_req.get(instance) is ledger:
                thread.ledger_by_req.pop(instance, None)
                thread.instance_to_req.pop(instance, None)
                thread.receipt_deadlines.pop(instance, None)
                thread.receipt_mode_reqs.discard(req_id)
                if not any(get_external_request_id(r) == external for r in thread.instance_to_req.values()):
                    thread.receipt_mode_reqs.discard(external)
            if self.request_map.get(req_id) == req_id:
                self.request_map.pop(req_id, None)
            if self.request_map.get(external) == req_id:
                self.request_map.pop(external, None)
            getattr(self, "_admitted_target_blocks", {}).pop(req_id, None)

    def _install_receipt_specs(self, specs: dict) -> None:
        """Build the D-side ReceiptLedger per request from the scheduler-built
        spec (cross-process transport via connector metadata). The
        receive thread stays the sole ledger writer."""
        assert self.kv_recv_layer_thread is not None
        thread = self.kv_recv_layer_thread
        for instance, spec in specs.items():
            req_id, expected_ranks, backings, prompt_len, lease = spec
            if isinstance(lease, TargetManifest):
                delivery = getattr(self, "_target_delivery", None)
                if delivery is None:
                    raise ContractError("Rank manifest arrived without owner receive admission")
                delivery.admit_target(lease, req_id)
                continue
            indexer_backings: tuple[str, ...] = ()
            if lease is not None:
                region_by_id = {region.region_id: region for region in lease.regions}
                indexer_backings = tuple(
                    dict.fromkeys(
                        region_by_id[view.region_binding].backing_id
                        for view in lease.views
                        if view.component is Component.INDEXER and view.region_binding in region_by_id
                    )
                )
            ledger = ReceiptLedger(
                request=RequestKey(request_instance=instance, transfer_generation=0),
                expected_ranks=expected_ranks,
                main_backings=backings,
                indexer_ranks=expected_ranks,
                prompt_len=prompt_len,
                required_layers=tuple(range(self.total_layers)),
                indexer_backings=indexer_backings,
            )
            if lease is not None:
                ledger.lease = lease  # consumed by prepare_decode on publish
            external_req_id = get_external_request_id(req_id)
            with thread.lock:
                thread.ledger_by_req[instance] = ledger
                thread.instance_to_req[instance] = external_req_id
                thread.receipt_mode_reqs.add(external_req_id)
                thread.receipt_deadlines[instance] = time.monotonic() + RECEIPT_WATCHDOG_SECONDS
            logger.debug(
                "Installed receipt ledger for req %s (instance=%s, expected=%s)", req_id, instance, expected_ranks
            )
        if self._target_provider is not None:
            thread.receipt_provider = self._target_provider

    def prepare_source_batch(self, source, requests, ready_event):
        """Build the complete local reader list without peer/native work."""
        metadata = self._source_request_metadata
        legacy_expected = any(meta.remote_target_manifest is None for meta in metadata.values())
        jobs = []
        by_id = {request.request_id: request for request in requests}
        for req_id, meta in metadata.items():
            if meta.remote_target_manifest is None:
                continue
            request = by_id.get(req_id)
            if request is None:
                raise ContractError("Admitted PD request is missing from the Store source batch")
            end = min(request.end_token, meta.remote_target_manifest.lease.allocation.prompt_tokens)
            start = max(request.start_token, min(meta.local_transed_tokens, end))
            job = self._manifest_source_writer.prepare_job(
                source, replace(request, start_token=start, end_token=end), ready_event
            )
            if job is not None:
                jobs.append(job)
        return tuple(jobs), legacy_expected

    def prepare_layerwise_step(self, metadata: MooncakeLayerwiseConnectorMetadata) -> None:
        """Prepare layerwise hooks once, when metadata is bound before forward."""
        self.current_layer = 0
        self._cache_write_events = [None] * self.total_layers
        if self.vllm_config.kv_transfer_config.is_kv_producer:
            self._source_request_metadata = dict(metadata.requests)
            # update trans info
            update_metadata = {}
            for req_idx, (req_id, req_meta) in enumerate(metadata.requests.items()):
                if req_meta.remote_target_manifest is not None:
                    if not self._source_contract_bound:
                        raise ContractError("Manifest producer requires a wired Store SourceReady registry")
                    if (
                        self.pcp_size != 1
                        or self.dcp_size != 1
                        or req_meta.remote_pcp_size != 1
                        or req_meta.remote_dcp_size != 1
                        or req_meta.remote_tp_size != self.tp_size
                        or self.pd_head_ratio != 1
                        or self.enable_c8_quant
                        or self.enable_kv_quant
                    ):
                        raise ContractError("Manifest SourceReady writes require equal TP, CP1 and BF16 Main")
                    self._manifest_source_writer.register_request(req_id, req_meta)
                    update_metadata[req_id] = req_meta
                    continue
                transfer_mappings: dict[tuple[str, int], dict[str, Any]] = {}
                self._align_remote_block_ids(req_meta)
                for i, kv_cache_spec in enumerate(self.kv_cache_specs):
                    if isinstance(kv_cache_spec, MambaSpec):
                        single_group_transfer_mappings = self._get_kv_split_metadata_for_mamba(
                            req_meta, req_idx, req_id, i
                        )
                    else:
                        single_group_transfer_mappings = self._get_kv_split_metadata(req_meta, req_idx, req_id, i)
                    for (host, port), block_dict in single_group_transfer_mappings.items():
                        if (host, port) not in transfer_mappings:
                            transfer_mappings[(host, port)] = {
                                "local_block_ids": [[] for _ in range(self.num_kv_cache_groups)],
                                "remote_block_ids": [[] for _ in range(self.num_kv_cache_groups)],
                                "trans_count": [0 for _ in range(self.num_kv_cache_groups)],
                            }
                        transfer_mappings[(host, port)]["local_block_ids"][i].extend(
                            single_group_transfer_mappings[(host, port)]["local_block_ids"]
                        )
                        transfer_mappings[(host, port)]["remote_block_ids"][i].extend(
                            single_group_transfer_mappings[(host, port)]["remote_block_ids"]
                        )
                        transfer_mappings[(host, port)]["trans_count"][i] = single_group_transfer_mappings[
                            (host, port)
                        ]["trans_count"]
                assert len(transfer_mappings) <= 1, f"Not support add mutil transfer task for req_id:{req_id}"
                update_req_meta = copy.deepcopy(req_meta)
                for (host, port), block_dict in transfer_mappings.items():
                    update_req_meta.remote_host = host
                    update_req_meta.remote_port = port
                    update_req_meta.local_block_ids = self._get_kernel_block_ids(block_dict["local_block_ids"])
                    update_req_meta.remote_block_ids = self._get_kernel_block_ids(block_dict["remote_block_ids"])
                    update_req_meta.trans_count = block_dict["trans_count"]
                    update_metadata[req_id] = update_req_meta
            metadata.requests = {}
            for req_id, req_meta in update_metadata.items():
                metadata.requests[req_id] = update_metadata[req_id]

            # update send task trans block info
            if self.attn_resharding_group_idx:
                send_task = metadata.send_task
                send_task.group_rearrange_block_ids = [[] for _ in range(self.num_kv_cache_groups)]
                send_task.group_num_blocks = [0 for _ in range(self.num_kv_cache_groups)]
                send_task.group_num_tokens = [0 for _ in range(self.num_kv_cache_groups)]
                send_task.group_block_table = [None for _ in range(self.num_kv_cache_groups)]
                send_task.group_block_len_tensor = [None for _ in range(self.num_kv_cache_groups)]
                send_task.group_seq_start_tensor = [None for _ in range(self.num_kv_cache_groups)]
                device = self.k_buffer.device  # type: ignore
                for i in self.attn_resharding_group_idx:
                    send_task.group_rearrange_block_ids[i].extend(
                        sorted(
                            {
                                block_id
                                for req_id in metadata.requests
                                for block_id in metadata.requests[req_id].local_block_ids[i]
                            }
                        )
                    )
                    flat_block_ids = send_task.group_rearrange_block_ids[i]
                    block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int32, device=device)
                    send_task.group_num_blocks[i] = len(flat_block_ids)
                    send_task.group_num_tokens[i] = send_task.group_num_blocks[i] * (
                        self.block_size[i] // self.kernel_block_size_scale[i]
                    )

                    send_task.group_block_table[i] = block_ids_tensor.view(1, -1)
                    send_task.group_block_len_tensor[i] = torch.tensor(
                        [send_task.group_num_tokens[i]], dtype=torch.int32, device=device
                    )
                    send_task.group_seq_start_tensor[i] = torch.tensor([0], dtype=torch.int32, device=device)

    def on_kv_cache_written(self, layer_name: str) -> None:
        if self.current_layer >= self.total_layers:
            return
        layer_names = self.index_to_name[self.current_layer]
        if not layer_names:
            raise RuntimeError(f"Mooncake layerwise metadata is missing physical layer {self.current_layer}")
        if layer_name and layer_name not in layer_names:
            raise RuntimeError(
                f"Mooncake layer hook {layer_name!r} does not belong to physical layer {self.current_layer}: "
                f"{layer_names}"
            )
        cache_write_event = torch.npu.Event()
        cache_write_event.record()
        self._cache_write_events[self.current_layer] = cache_write_event

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: list[torch.Tensor],
        attn_metadata: "AttentionMetadata",
        connector_metadata: MooncakeLayerwiseConnectorMetadata,
        **kwargs,
    ) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        if self.vllm_config.kv_transfer_config.is_kv_producer and connector_metadata.requests.keys():
            if self.current_layer >= self.total_layers:
                self.current_layer += 1
                return
            layer_names = self.index_to_name[self.current_layer]
            if not layer_names:
                raise RuntimeError(f"Mooncake layerwise metadata is missing physical layer {self.current_layer}")
            # Use the scatter-time event recorded by on_kv_cache_written.
            # The fallback preserves correctness for attention paths that do
            # not call the early hook, but gives up transfer/compute overlap.
            if layer_name == "":
                layer_name = layer_names[0]
            elif layer_name not in layer_names:
                raise RuntimeError(
                    f"Mooncake layer hook {layer_name!r} does not belong to physical layer {self.current_layer}: "
                    f"{layer_names}"
                )
            cache_write_event = self._cache_write_events[self.current_layer]
            self._cache_write_events[self.current_layer] = None
            if cache_write_event is None:
                cache_write_event = torch.npu.Event()
                cache_write_event.record()
            # compute-done event, recorded AFTER this hook returns so
            # attention has finished reading. Semantically distinct from
            # _cache_write_events (the scatter-time ready event handed to the
            # transfer thread) — do not merge them. A layer whose requests were
            # all fully prefix-hit has no compute reader mark it
            # no_compute_reader instead of recording an event. The predicate is
            # pure stdlib (receipt module) so the receipt unit tests cover it on any box.
            if is_no_compute_layer(
                [r.local_computed_tokens for r in connector_metadata.requests.values()],
                [r.prompt_len for r in connector_metadata.requests.values()],
            ):
                self._no_compute_layers.add(self.current_layer)
            elif self.current_layer not in self._no_compute_layers:
                compute_done_event = torch.npu.Event()
                compute_done_event.record()
                self._compute_done_events[self.current_layer] = compute_done_event
            send_task = connector_metadata.send_task
            layer_group_idx = self.layer_metadata[layer_name].tensor_group_idx[0]
            keys = None
            values = None
            quant_keys = None
            quant_values = None
            if layer_name not in self.sparse_sfa_c8_layers and (
                (
                    self.pd_head_ratio != 1
                    and (isinstance(self.kv_cache_specs[layer_group_idx], (FullAttentionSpec, SlidingWindowSpec)))
                    and send_task.group_num_blocks[layer_group_idx] > 0
                )
                or (self.enable_c8_quant and self.current_layer in self.vllm_config.quant_config.c8_quant_layers)
                or (self.enable_kv_quant and self.current_layer in self.vllm_config.quant_config.kvcache_quant_layers)
            ):
                assert self.resharding_stream is not None
                with npu_stream_switch(self.resharding_stream):
                    cache_write_event.wait()
                    device = self.k_buffer.device  # type: ignore
                    # Initialize buffers
                    keys = torch.empty(
                        (send_task.group_num_tokens[layer_group_idx], *kv_layer[0].size()[-2:]),
                        dtype=kv_layer[0].dtype,
                        device=device,
                    )
                    values = torch.empty(
                        (send_task.group_num_tokens[layer_group_idx], *kv_layer[1].size()[-2:]),
                        dtype=kv_layer[1].dtype,
                        device=device,
                    )

                    # Load cache data into buffers
                    torch_npu.npu_gather_pa_kv_cache(
                        kv_layer[0],
                        kv_layer[1],
                        send_task.group_block_table[layer_group_idx],
                        send_task.group_block_len_tensor[layer_group_idx],
                        seq_offset=send_task.group_seq_start_tensor[layer_group_idx],
                        key=keys,
                        value=values,
                    )
                    if self.pd_head_ratio != 1:
                        # sort kv caches for each block
                        keys = (
                            keys.view(
                                send_task.group_num_blocks[layer_group_idx], self.pd_head_ratio, -1, *keys.shape[1:]
                            )
                            .transpose(0, 1)
                            .reshape_as(keys)
                        )
                        values = (
                            values.view(
                                send_task.group_num_blocks[layer_group_idx], self.pd_head_ratio, -1, *values.shape[1:]
                            )
                            .transpose(0, 1)
                            .reshape_as(values)
                        )
                        # reshard kv cache
                        keys = keys.reshape(-1, *kv_layer[0].shape[2:])
                        values = values.reshape(-1, *kv_layer[1].shape[2:])

                        (keys, values) = kv_alltoall_and_rearrange(self.pd_head_ratio, keys, values)
                    if self.enable_c8_quant:
                        layer = self.vllm_config.compilation_config.static_forward_context[layer_name]
                        quant_keys = torch.clamp(
                            torch.round(keys * layer._c8_k_inv_scale + layer._c8_k_offset),
                            -128,
                            127,
                        ).to(torch.int8)
                        quant_values = torch.clamp(
                            torch.round(values * layer._c8_v_inv_scale + layer._c8_v_offset),
                            -128,
                            127,
                        ).to(torch.int8)
                        quant_keys = self.get_nz_cache(quant_keys, layer_group_idx)
                        quant_values = self.get_nz_cache(quant_values, layer_group_idx)
                    if (
                        self.enable_kv_quant
                        and self.current_layer in self.vllm_config.quant_config.kvcache_quant_layers
                    ):
                        layer = self.vllm_config.compilation_config.static_forward_context[layer_name]
                        keys = torch.ops.vllm.quantize(
                            keys, layer.fak_descale, layer.fak_descale_reciprocal, layer.fak_offset
                        )
                        quant_keys = self.get_nz_cache(keys, layer_group_idx)
                        quant_values = self.get_nz_cache(values, layer_group_idx)

            assert self.kv_send_layer_thread is not None
            layer_send_task = SendTask(
                wait_event=cache_write_event,
                k_cache=keys,
                v_cache=values,
                k_quant_cache=quant_keys,
                v_quant_cache=quant_values,
                layer_idx=self.current_layer,
                layer_name=layer_name,
                layer_names=layer_names,
                group_rearrange_block_ids=send_task.group_rearrange_block_ids,
            )
            for req_id, req_meta in connector_metadata.requests.items():
                if len(req_meta.local_block_ids) <= layer_group_idx or not req_meta.local_block_ids[layer_group_idx]:
                    continue
                if req_meta.remote_target_manifest is not None:
                    continue  # The SourceReady job is this request's sole data-plane reader.
                # Apply the lease writer plan: a Main backing has exactly one
                # writer rank; other ranks skip it to avoid duplicate sends.
                # Indexer ranges go to every consuming rank. v1 single-backing:
                # the plan's writers_by_backing has one Main entry.
                writer_plan = getattr(req_meta, "remote_writer_plan", None)
                if writer_plan is not None and isinstance(
                    self.kv_cache_specs[layer_group_idx], (FullAttentionSpec, SlidingWindowSpec)
                ):
                    main_writers = tuple(
                        ranks for backing, ranks in writer_plan.writers_by_backing.items() if ranks and len(ranks) == 1
                    )
                    if main_writers and self.tp_rank not in main_writers[0]:
                        continue
                try:
                    req_meta_update = self.update_decoder_info(req_id, req_meta)
                except Exception as e:
                    logger.warning(
                        "MooncakeLayerwiseConnector transfer fail. req_id=%s, layer_idx=%s, error=%s. ",
                        req_id,
                        self.current_layer,
                        e,
                    )
                    layer_send_task.failed_requests[req_id] = req_meta
                    continue
                logger.debug("Add request %s to kv send layer thread. req_meta_update=%r", req_id, req_meta_update)
                layer_send_task.send_request[req_id] = req_meta_update

            needs_reuse_gate = any(
                len(req_meta.local_block_ids) > layer_group_idx and bool(req_meta.local_block_ids[layer_group_idx])
                for req_meta in connector_metadata.requests.values()
                if req_meta.remote_target_manifest is None
            )
            # register the layer's whole batch — all
            # requests — in one atomic register+seal, so a shared layer cannot
            # release its slot on the first finished request. Receipt-mode
            # requests only; plain-mode layers keep the legacy event path, and
            # unregistered layers fall through to it as well.
            if self._slot_ledger is not None and layer_send_task.send_request:
                receipt_keys = tuple(
                    self._build_transfer_key(req_id, req_meta, self.current_layer)
                    for req_id, req_meta in layer_send_task.send_request.items()
                    if req_meta.request_instance
                )
                if receipt_keys:
                    try:
                        self._slot_ledger.register_batch(self.current_layer, receipt_keys)
                    except SlotError as slot_error:
                        raise RuntimeError(
                            f"Mooncake layerwise slot ledger rejected layer {self.current_layer}: {slot_error}"
                        ) from slot_error
                    self._layer_keys[self.current_layer] = receipt_keys
                    self._slot_ledger_active = True
            if needs_reuse_gate:
                # This must happen before AscendStore publishes StoreSaveDone.
                # Otherwise its receiver may consume the previous set state
                # and start H2D before this transfer is registered as pending.
                self._mark_layer_reuse_pending(self.current_layer)
            try:
                # When the source contract is active, hand the assembled task to
                # the consumer so it is released through source_ready (readers
                # registered before the async send runs). Plain mode, or a layer
                # the registry did not open, enqueues directly, preserving the
                # legacy path. hand_off_send_task itself falls back to a direct
                # enqueue when no open generation exists for the layer.
                consumer = getattr(self, "_reuse_source_consumer", None)
                if consumer is not None:
                    consumer.hand_off_send_task(self.current_layer, layer_send_task)
                else:
                    self.kv_send_layer_thread.send_queue.put(layer_send_task)
            except Exception as error:
                if needs_reuse_gate:
                    self._complete_layer_reuse(self.current_layer, str(error))
                raise
            self.current_layer += 1

    # NOTE: Due to the FIA operator constraints, the expected kv cache is ND format, NZ shape,
    # while the npu_format_cast method only modifies the memory layout, we manually convert it to NZ shape here
    def get_nz_cache(self, cache_tensor: torch.Tensor, layer_group_idx: int):
        head_num, head_dim = cache_tensor.shape[-2], cache_tensor.shape[-1]
        cache_tensor = cache_tensor.view(-1, self.block_size[layer_group_idx], head_num * head_dim)
        cache_tensor = trans_nd_to_nz(cache_tensor)
        cache_tensor = cache_tensor.reshape(-1, head_num, head_dim)
        return cache_tensor

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()

            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(
                ctx=ctx,
                path=remote_path,
                socket_type=zmq.REQ,  # type: ignore
                bind=False,
            )
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            return sock

    def update_decoder_info(self, req_id, req_meta: ReqMeta):
        if (
            req_meta.remote_engine_id not in self.remote_layer_metadata
            or req_meta.remote_port not in self.remote_layer_metadata[req_meta.remote_engine_id]
        ):
            try:
                encoded_data = self.encoder.encode((GET_META_MSG, req_id))
                sock = self._get_remote_socket(req_meta.remote_host, req_meta.remote_port)
                path = f"{req_meta.remote_host}:{req_meta.remote_port}"
                ensure_zmq_send(sock, encoded_data, path)
                metadata_bytes = ensure_zmq_recv(sock, self.remote_poller, path)
                agent_meta: MooncakeAgentMetadata = self.decoder.decode(metadata_bytes)
            except Exception as e:
                logger.error(
                    "Query to port and kv base addr for request fail. req_id=%s, source=%s:%s, error=%s. ",
                    req_id,
                    req_meta.remote_host,
                    req_meta.remote_port,
                    e,
                )
                raise e
            assert req_meta.remote_engine_id != self.engine_id, (
                f"Conflict engine id {req_meta.remote_engine_id} with local engine id {self.local_engine_id}."
            )
            self.remote_layer_metadata[req_meta.remote_engine_id][req_meta.remote_port] = agent_meta.layer_metadata
            self.remote_te_port[req_meta.remote_engine_id][req_meta.remote_port] = agent_meta.te_rpc_port
            logger.debug(
                "Query to port and kv base addr for request %s from %s:%s success "
                "agent_meta.layer_metadata=%r agent_meta.te_rpc_port=%r",
                req_id,
                req_meta.remote_host,
                req_meta.remote_port,
                agent_meta.layer_metadata,
                agent_meta.te_rpc_port,
            )
            if self.pd_head_ratio > 1:
                # for tp inequal, pre-create link to prevent alltoall out of memory
                session_id = f"{req_meta.remote_host}:{agent_meta.te_rpc_port}"
                first_layer_name = next(iter(self.layer_metadata.keys()))
                ret = self.engine.batch_transfer_sync_write(
                    session_id,
                    [self.layer_metadata[first_layer_name].kv_caches_base_addr[0]],
                    [agent_meta.layer_metadata[first_layer_name].kv_caches_base_addr[0]],
                    [128],
                )
                if ret < 0:
                    logger.error("Mooncake transfer failed to create link. session_id=%s, ret=%d. ", session_id, ret)
        req_meta.remote_te_rpc_port = self.remote_te_port[req_meta.remote_engine_id][req_meta.remote_port]
        req_meta.remote_layer_metadata = self.remote_layer_metadata[req_meta.remote_engine_id][req_meta.remote_port]
        return req_meta

    def send_done_send_signal(self, req_id, req_meta, group_idx, trans_flag: bool = True):
        external_req_id = get_external_request_id(req_id)
        send_msg_type = DONE_SENDING_MSG if trans_flag else FAILED_SENDING_MSG
        logger.info(
            "Sending transmitting signal %s for request %s to %s:%d",
            send_msg_type,
            external_req_id,
            req_meta.remote_host,
            req_meta.remote_port,
        )
        try:
            path = make_zmq_path("tcp", req_meta.remote_host, req_meta.remote_port)
            msg_encoder = msgspec.msgpack.Encoder()
            side_channel_path = f"{self.side_channel_host}:{self.handshake_port}"
            if req_meta.request_instance:
                # Receipt-mode request: version-tagged 5-tuple. Plain requests
                # keep the 4-tuple, so a plain-mode D side reads it unchanged
                # (the extra element is simply never looked at there).
                encoded_data = msg_encoder.encode(
                    (
                        send_msg_type,
                        external_req_id,
                        req_meta.trans_count[group_idx],
                        side_channel_path,
                        CONTRACT_VERSION,
                    )
                )
            else:
                encoded_data = msg_encoder.encode(
                    (send_msg_type, external_req_id, req_meta.trans_count[group_idx], side_channel_path)
                )
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore
                        sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
                        ensure_zmq_send(sock, encoded_data, f"{req_meta.remote_host}:{req_meta.remote_port}")
                        if not sock.poll(int(self.timeout * 1000), zmq.POLLIN):  # type: ignore[attr-defined]
                            raise TimeoutError(
                                f"Timed out waiting for ACK from {req_meta.remote_host}:{req_meta.remote_port}"
                            )
                        ack = sock.recv()
                        if ack != b"ACK":
                            raise ValueError(f"Unexpected ACK response: {ack}")
                        return
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(
                            "Failed to send done sending signal. "
                            "request_id=%s, destination=%s:%d, attempt=%d/%d, error=%s. ",
                            external_req_id,
                            req_meta.remote_host,
                            req_meta.remote_port,
                            attempt,
                            max_retries,
                            e,
                        )
                        time.sleep(0.1)
                    else:
                        raise RuntimeError(f"Failed to receive ACK after {max_retries} attempts: {e}") from e
        except Exception as e:
            logger.error(
                "Sending signal fail. signal_type=%s, request_id=%s, destination=%s:%s, error=%s. ",
                send_msg_type,
                external_req_id,
                req_meta.remote_host,
                req_meta.remote_port,
                e,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:  # type: ignore
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def group_concurrent_contiguous(
    src: list[int], dst: list[int] | None = None
) -> tuple[list[npt.NDArray[np.int64]], list[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    if dst is None:
        dst = []
    if not dst:
        src_only_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)

        if src_only_indices.size == 0:
            return [], []

        brk = np.where(np.diff(src_only_indices) != 1)[0] + 1
        src_groups = np.split(src_only_indices, brk)
        src_groups = [g.tolist() for g in src_groups]

        return src_groups, []

    else:
        src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
        dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

        if src_indices.size == 0:
            return [], []

        brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
        src_groups = np.split(src_indices, brk)
        dst_groups = np.split(dst_indices, brk)

        src_groups = [g.tolist() for g in src_groups]
        dst_groups = [g.tolist() for g in dst_groups]

        return src_groups, dst_groups


def string_to_int64_hash(input_str):
    """
    Hash the string using SHA-256 and convert it into an int64 integer.
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def ensure_zmq_send(
    socket: zmq.Socket,  # type: ignore
    data: bytes,
    path: str,
    max_retries: int = 3,
):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Send failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Send failed after all retries. error=%s. ", e)
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def ensure_zmq_recv(
    socket: zmq.Socket,  # type: ignore
    poller: zmq.Poller,  # type: ignore
    path: str,
    timeout: float = 1.0,
    max_retries: int = 3,
) -> bytes:
    retries_left = max_retries
    while True:
        try:
            if dict(poller.poll(int(timeout * 1000))):  # milliseconds
                data = socket.recv()
                return data
            else:
                raise zmq.ZMQError("Receive timeout")  # type: ignore
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Receive failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Receive failed after all retries. error=%s. ", e)
                raise RuntimeError(f"Failed to receive data from {path} after {max_retries} retries: {e}")


def get_external_request_id(request_id: str):
    # NOTE(zxr): vLLM PR #27987 add additional suffix
    # to EngineCore request_id with len(suffix) == 9
    return request_id[:-9]

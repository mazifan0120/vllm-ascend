# SPDX-License-Identifier: Apache-2.0
"""Scheduler integration acceptance for Task-01 Decode admission (spec §11.4).

Uses the real vLLM v1 Scheduler on CPU with the real ``DualPathConnector``;
only the KVPool backend module and the ``LookupKeyClient`` transport are
constrained at their existing seams. Proves that for ``L_DE < R`` one
``schedule()`` call admits the request into ``WAITING_FOR_REMOTE_KVS`` with
final blocks bound in a ``DecodeKVSnapshot``, and that an HBM-complete
request takes the normal local path with no Task-01 state.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.config import (  # noqa: E402
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

from vllm_ascend.distributed.kv_transfer import register_connector  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import (  # noqa: E402
    DualPathConnector,
    DualPathConnectorScheduler,
)

_BLOCK_SIZE = 16
_NONE_HASH_INITIALIZED = False
_CONNECTOR_REGISTERED = False


def _ensure_connector_registered() -> None:
    global _CONNECTOR_REGISTERED
    if not _CONNECTOR_REGISTERED:
        register_connector()
        _CONNECTOR_REGISTERED = True


def _make_vllm_config() -> VllmConfig:
    fake_weight_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "_fake_weight")
    model_config = ModelConfig(model=fake_weight_path, skip_tokenizer_init=True)
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=1024,
        max_model_len=1024,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=_BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="DualPathConnector",
        kv_role="kv_consumer",
        kv_connector_extra_config={
            "role": "decode",
            "consumer_is_to_load": True,
            "backend": "mooncake",
            "lookup_rpc_port": 18883,
        },
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


def _make_scheduler(vllm_config: VllmConfig, num_blocks: int = 1000) -> Scheduler:
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size=_BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float16),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        block_size=_BLOCK_SIZE,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _make_request(request_id: str, prompt_token_ids: list[int], kv_transfer_params: dict | None) -> Request:
    global _NONE_HASH_INITIALIZED
    if not _NONE_HASH_INITIALIZED:
        init_none_hash(sha256)
        _NONE_HASH_INITIALIZED = True
    sampling_params = SamplingParams(max_tokens=8)
    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(_BLOCK_SIZE, sha256),
    )
    request.kv_transfer_params = kv_transfer_params
    return request


def _runner_output_for(requests: list[Request]) -> ModelRunnerOutput:
    req_ids = [request.request_id for request in requests]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[[0] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=KVConnectorOutput(finished_sending=set(), finished_recving=set()),
    )


@pytest.fixture(autouse=True)
def _constrain_kvpool_seams():
    """Constrain the KVPool backend resolution and lookup transport at their
    existing seams; everything between vLLM core and the adapter stays real."""
    _ensure_connector_registered()
    with (
        patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib") as mock_importlib,
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"
        ) as mock_lookup_client_cls,
    ):
        mock_importlib.import_module.return_value = MagicMock()
        yield mock_lookup_client_cls


@pytest.fixture()
def scheduler():
    instance = _make_scheduler(_make_vllm_config())
    yield instance
    instance.shutdown()


def _dual_scheduler(scheduler: Scheduler) -> DualPathConnectorScheduler:
    connector = scheduler.connector
    assert isinstance(connector, DualPathConnector)
    dual = connector.connector_scheduler
    assert isinstance(dual, DualPathConnectorScheduler)
    return dual


def _admit_one_request(scheduler: Scheduler):
    """Schedule one DualPath Decode request (P=33, R=32, L_DE=0) and return the
    request, the SchedulerOutput, the matched-token return capture, and the
    lookup/allocate spies of the admission step."""
    dual = _dual_scheduler(scheduler)
    matched_returns: list[tuple[int, bool]] = []
    real_matched = scheduler.connector.get_num_new_matched_tokens

    def capture_matched(request, num_computed_tokens):
        result = real_matched(request, num_computed_tokens)
        matched_returns.append(result)
        return result

    request = _make_request("req-de", list(range(33)), {"do_remote_prefill": True})
    scheduler.add_request(request)
    with (
        patch.object(dual._kvpool_adapter, "lookup", wraps=dual._kvpool_adapter.lookup) as lookup_mock,
        patch.object(scheduler.connector, "get_num_new_matched_tokens", side_effect=capture_matched) as matched_mock,
        patch.object(
            scheduler.kv_cache_manager, "allocate_slots", wraps=scheduler.kv_cache_manager.allocate_slots
        ) as alloc_mock,
    ):
        scheduler_output = scheduler.schedule()
    return request, scheduler_output, matched_returns, lookup_mock, matched_mock, alloc_mock


def _assert_admission_invariants(scheduler, request, scheduler_output, matched_returns, alloc_mock, expect_store_spec):
    dual = _dual_scheduler(scheduler)
    # 1. connector returned (R - L_DE, True) = (32, True)
    assert matched_returns == [(32, True)]
    # 2. allocate_slots received the external delta with delayed caching
    assert alloc_mock.call_args.kwargs["num_external_computed_tokens"] == 32
    assert alloc_mock.call_args.kwargs["delay_cache_blocks"] is True
    # 3. final block IDs exist and equal the snapshot's frozen IDs
    final_block_ids = tuple(
        tuple(group) for group in scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
    )
    snapshot = dual._decode_kv_snapshots[request.request_id]
    assert snapshot.final_block_ids == final_block_ids
    assert snapshot.target_tokens == 32
    assert snapshot.external_tokens == 32
    if expect_store_spec:
        assert snapshot.store_load_spec is not None
    else:
        assert snapshot.store_load_spec is None
    # 4. the request waits for remote KVs
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    # 5. vLLM recorded the Decode-ready token count for the pending receive
    assert request.num_computed_tokens == 32
    # 6. no model tokens were scheduled for the request in this step
    assert scheduler_output.num_scheduled_tokens.get(request.request_id, 0) == 0
    # 7. connector metadata contains no Store or P2P work for the request
    metadata = scheduler_output.kv_connector_metadata
    assert metadata is None or request.request_id not in metadata.requests
    assert dual._reqs_need_recv == {}
    # 8. no finished_recving completion is published
    scheduler.update_from_output(scheduler_output, _runner_output_for([]))
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.finished_recving_kv_req_ids == set()


def test_admission_full_store_hit(_constrain_kvpool_seams, scheduler):
    _constrain_kvpool_seams.return_value.lookup.return_value = 32
    request, scheduler_output, matched_returns, lookup_mock, _, alloc_mock = _admit_one_request(scheduler)
    lookup_mock.assert_called_once()
    _assert_admission_invariants(
        scheduler, request, scheduler_output, matched_returns, alloc_mock, expect_store_spec=True
    )
    snapshot = _dual_scheduler(scheduler)._decode_kv_snapshots[request.request_id]
    assert snapshot.store_load_spec.kvpool_cached_tokens == 32
    assert snapshot.store_tokens == 32


def test_admission_partial_store_hit(_constrain_kvpool_seams, scheduler):
    _constrain_kvpool_seams.return_value.lookup.return_value = 16
    request, scheduler_output, matched_returns, lookup_mock, _, alloc_mock = _admit_one_request(scheduler)
    lookup_mock.assert_called_once()
    _assert_admission_invariants(
        scheduler, request, scheduler_output, matched_returns, alloc_mock, expect_store_spec=True
    )
    snapshot = _dual_scheduler(scheduler)._decode_kv_snapshots[request.request_id]
    # Partial hit does not change the Core-facing external delta.
    assert snapshot.store_load_spec.kvpool_cached_tokens == 16
    assert snapshot.external_tokens == 32


def test_admission_store_miss(_constrain_kvpool_seams, scheduler):
    _constrain_kvpool_seams.return_value.lookup.return_value = 0
    request, scheduler_output, matched_returns, lookup_mock, _, alloc_mock = _admit_one_request(scheduler)
    lookup_mock.assert_called_once()
    _assert_admission_invariants(
        scheduler, request, scheduler_output, matched_returns, alloc_mock, expect_store_spec=False
    )


def test_hbm_complete_schedules_normally_without_task01_state(_constrain_kvpool_seams, scheduler):
    dual = _dual_scheduler(scheduler)
    prompt = list(range(33))

    # Prime the prefix cache with an identical ordinary request.
    primer = _make_request("req-primer", prompt, None)
    scheduler.add_request(primer)
    primer_output = scheduler.schedule()
    assert primer_output.num_scheduled_tokens[primer.request_id] == 33
    scheduler.update_from_output(primer_output, _runner_output_for([primer]))
    scheduler.finish_requests([primer.request_id], RequestStatus.FINISHED_STOPPED)

    lookup_spy = patch.object(dual._kvpool_adapter, "lookup", wraps=dual._kvpool_adapter.lookup)
    request = _make_request("req-de-hbm", prompt, {"do_remote_prefill": True})
    scheduler.add_request(request)
    with lookup_spy as lookup_mock:
        scheduler_output = scheduler.schedule()

    # L_DE = 32 = R: no KVPool lookup, no Task-01 state, normal local scheduling
    # with last-token recomputation.
    lookup_mock.assert_not_called()
    assert dual._lookup_results == {}
    assert dual._decode_kv_snapshots == {}
    assert scheduler_output.num_scheduled_tokens[request.request_id] == 1
    assert request.status == RequestStatus.RUNNING

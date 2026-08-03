# SPDX-License-Identifier: Apache-2.0
"""PR-00 NPU foundation parity smoke test from the foundation specification §12.

This test requires the NPU multi-node nightly runner. The exact command, image,
model, topology, and result are recorded in ``.specs/dual-path-stage1/TRACKING.md``.
Run on that runner with:

``pytest -sv tests/e2e/nightly/multi_node/dual_path/test_foundation_parity.py``

Evidence division: pairwise matched-token values, exactly-once Transfer Engine
and KV-buffer registration, and constructor thread counts are proven by the
CPU construction/behavior parity tests in
``tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py``
(spec section 11). This smoke test verifies the section 12 bullets that are
observable from a running disaggregated deployment. Every node captures only
its local server's stdout, so each node compares its own baseline run against
its own candidate run with role-appropriate markers: scheduler/worker
initialization and KV-buffer registration sequences on every node, and
recv-thread startup, remote-prefill admission, and transfer-completion counts
on the decode (kv_consumer) node where those parent log lines exist. The
master (prefill) node additionally compares token-identical outputs and
request-terminal outcomes through the proxy API. All log markers are verified
against real parent log lines in mooncake_layerwise_connector.py; markers
without a guaranteed log line are deliberately not used.
"""

import asyncio
import signal
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.multi_node.internal_dp.scripts.multi_node_config import (
    MultiNodeConfig,
    MultiNodeConfigLoader,
    ProxyLauncher,
)

BASELINE_CONFIG_PATH: Final = (
    "tests/e2e/nightly/multi_node/dual_path/config/"
    "GLM-4.7-W8A8C8-Mooncake-Layerwise-baseline.yaml"
)
CANDIDATE_CONFIG_PATH: Final = (
    "tests/e2e/nightly/multi_node/dual_path/config/"
    "GLM-4.7-W8A8C8-DualPath-parity.yaml"
)

FIXED_PROMPTS: Final[list[str]] = [
    "What is two plus three?",
    "Name the largest planet in our solar system.",
    "Give one synonym for quick.",
    "What color is a clear daytime sky?",
    "Complete the sequence: 2, 4, 6,",
    "Name the author of Hamlet.",
]
REQUEST_SEED: Final = 1024
MAX_TOKENS: Final = 64
REQUEST_TIMEOUT_SECONDS: Final = 300.0
BETWEEN_RUN_SETTLE_SECONDS: Final = 5.0

# Real parent log lines (mooncake_layerwise_connector.py): scheduler init
# (:798), worker init (:1135 and :1171 — the parent logs it twice per worker,
# so worker-init counts are asserted > 0 on both runs but never for equality),
# consumer recv-thread startup (:607), remote-prefill admission (:953), and
# transfer-level recv completion (:1425).
BASELINE_SCHEDULER_INIT_MARKERS: Final[tuple[str, ...]] = ("Initializing Mooncake Scheduler",)
CANDIDATE_SCHEDULER_INIT_MARKERS: Final[tuple[str, ...]] = ("Initializing DualPath Scheduler",)
BASELINE_WORKER_INIT_MARKERS: Final[tuple[str, ...]] = ("Initializing Mooncake work",)
CANDIDATE_WORKER_INIT_MARKERS: Final[tuple[str, ...]] = ("Initializing DualPath Worker",)
RECV_THREAD_MARKERS: Final[tuple[str, ...]] = ("KVCacheRecvingLayerThread listening on",)
ADMISSION_MARKERS: Final[tuple[str, ...]] = ("Send request:",)
COMPLETION_MARKERS: Final[tuple[str, ...]] = ("Number of completed KV cache recv requests",)
REGISTRATION_MARKERS: Final[tuple[str, ...]] = ("num_blocks: ",)
CANDIDATE_NEGATIVE_MARKERS: Final[tuple[str, ...]] = (
    "AscendStore",
    "StoreConnector",
    "PathDecision",
    "round-robin",
    "round_robin",
    "Reverse transfer",
    "DE_PARTIAL_HIT",
    "DE_LOCAL_FULL_HIT",
)
FAILURE_MARKERS: Final[tuple[str, ...]] = (
    "Traceback",
    "load errors",
    "invalid block",
)


class _FrozenPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _MessagePayload(_FrozenPayload):
    content: str


class _ChoicePayload(_FrozenPayload):
    message: _MessagePayload
    finish_reason: str


class _UsagePayload(_FrozenPayload):
    completion_tokens: int


class _ChatCompletionPayload(_FrozenPayload):
    choices: tuple[_ChoicePayload, ...]
    usage: _UsagePayload


@dataclass(frozen=True, slots=True)
class _CompletionRecord:
    prompt: str
    content: str
    finish_reason: str
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class _MarkerCounts:
    scheduler_initializations: int
    worker_initializations: int
    registrations: int
    recv_threads: int
    admissions: int
    transfer_completions: int


@dataclass(frozen=True, slots=True)
class _DeploymentResult:
    completions: tuple[_CompletionRecord, ...]
    worker_log: str
    is_master: bool
    is_decoder: bool
    server_was_running: bool
    server_returncode: int | None


def _count_markers(normalized_text: str, markers: tuple[str, ...]) -> int:
    return sum(normalized_text.count(marker.casefold()) for marker in markers)


def _extract_marker_counts(
    text: str,
    *,
    scheduler_init_markers: tuple[str, ...],
    worker_init_markers: tuple[str, ...],
) -> _MarkerCounts:
    normalized_text = text.casefold()
    return _MarkerCounts(
        scheduler_initializations=_count_markers(normalized_text, scheduler_init_markers),
        worker_initializations=_count_markers(normalized_text, worker_init_markers),
        registrations=_count_markers(normalized_text, REGISTRATION_MARKERS),
        recv_threads=_count_markers(normalized_text, RECV_THREAD_MARKERS),
        admissions=_count_markers(normalized_text, ADMISSION_MARKERS),
        transfer_completions=_count_markers(normalized_text, COMPLETION_MARKERS),
    )


def _assert_node_log_parity(baseline_log: str, candidate_log: str, *, is_decoder: bool) -> None:
    """Compare this node's own baseline-run log against its own candidate-run log.

    Each node's pytest process captures only its local server's stdout, and the
    consumer-side markers (recv thread, admission, transfer completion) exist
    only on the decode node, so every node asserts the markers appropriate to
    its configured role. Cross-run counts are comparable because both runs use
    the identical process topology on the same node.
    """
    baseline_counts = _extract_marker_counts(
        baseline_log,
        scheduler_init_markers=BASELINE_SCHEDULER_INIT_MARKERS,
        worker_init_markers=BASELINE_WORKER_INIT_MARKERS,
    )
    candidate_counts = _extract_marker_counts(
        candidate_log,
        scheduler_init_markers=CANDIDATE_SCHEDULER_INIT_MARKERS,
        worker_init_markers=CANDIDATE_WORKER_INIT_MARKERS,
    )

    # §12: the parent runtime initializes once per role on both runs.
    assert baseline_counts.scheduler_initializations > 0
    assert candidate_counts.scheduler_initializations > 0
    assert baseline_counts.scheduler_initializations == candidate_counts.scheduler_initializations
    # The parent logs the worker init line twice per worker (:1135 and :1171)
    # while the DualPath subclass logs it once, so worker-init counts are
    # asserted present on both runs but never compared for equality.
    assert baseline_counts.worker_initializations > 0
    assert candidate_counts.worker_initializations > 0
    # §12: one KV-buffer registration sequence per Worker on both runs.
    assert baseline_counts.registrations > 0
    assert baseline_counts.registrations == candidate_counts.registrations

    if not is_decoder:
        # The producer's send thread has no startup log line; its runtime
        # parity follows from the identical init/registration sequences above.
        return

    # §12: a real remote-prefill transfer must be exercised in both runs.
    assert baseline_counts.admissions > 0, "baseline logs show no remote-prefill admission"
    assert candidate_counts.admissions > 0, "candidate logs show no remote-prefill admission"
    # §12: identical matched-token outcomes at request granularity. The
    # matched-token value itself cannot be logged without modifying the
    # parent connector (forbidden by §14); its pairwise value equality is
    # proven by the CPU behavior-parity tests (§11.3), and the NPU smoke
    # verifies the observable admission outcome of every remote prefill.
    assert baseline_counts.admissions == candidate_counts.admissions
    # §12: identical request-terminal outcomes at transfer level.
    assert baseline_counts.transfer_completions > 0
    assert baseline_counts.transfer_completions == candidate_counts.transfer_completions
    # §12: no extra send/receive thread relative to the baseline role.
    assert baseline_counts.recv_threads > 0
    assert baseline_counts.recv_threads == candidate_counts.recv_threads


def _assert_markers_absent(text: str, markers: tuple[str, ...], run_name: str) -> None:
    normalized_text = text.casefold()
    present_markers = tuple(marker for marker in markers if marker.casefold() in normalized_text)
    assert not present_markers, f"{run_name} logs contain forbidden markers: {present_markers}"


async def _request_fixed_completions(endpoint: str, model: str) -> tuple[_CompletionRecord, ...]:
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    headers = {"Authorization": f"Bearer {RemoteOpenAIServer.DUMMY_API_KEY}"}
    completions: list[_CompletionRecord] = []

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        for prompt in FIXED_PROMPTS:
            response = await client.post(
                endpoint,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "seed": REQUEST_SEED,
                    "max_tokens": MAX_TOKENS,
                    "stream": False,
                },
            )
            response.raise_for_status()
            payload = _ChatCompletionPayload.model_validate(response.json())
            assert len(payload.choices) == 1
            choice = payload.choices[0]
            completions.append(
                _CompletionRecord(
                    prompt=prompt,
                    content=choice.message.content,
                    finish_reason=choice.finish_reason,
                    completion_tokens=payload.usage.completion_tokens,
                )
            )

    return tuple(completions)


async def _deploy_and_collect(
    config_path: str,
    capfd: pytest.CaptureFixture[str],
) -> _DeploymentResult:
    config: MultiNodeConfig = MultiNodeConfigLoader.from_yaml(config_path)
    completions: tuple[_CompletionRecord, ...] = ()

    with ProxyLauncher(
        nodes=config.nodes,
        disagg_cfg=config.disagg_cfg,
        envs=config.envs,
        proxy_port=config.proxy_port,
        cur_index=config.cur_index,
    ) as proxy:
        server = RemoteOpenAIServer(
            model=config.model,
            vllm_serve_args=config.server_cmd,
            server_port=config.server_port,
            server_host=config.master_ip,
            env_dict=config.envs,
            auto_port=False,
            proxy_port=proxy.proxy_port,
            disaggregated_prefill=config.disagg_cfg,
            nodes_info=config.nodes,
            max_wait_seconds=2800,
        )
        with server:
            server_was_running = server.proc.returncode is None
            if config.is_master:
                # RemoteOpenAIServer.__enter__ already waits for the proxy at
                # /healthcheck and for every api_server node (conftest.py), so
                # the deployment is ready to serve when the context opens.
                completions = await _request_fixed_completions(
                    f"http://{config.master_ip}:{proxy.proxy_port}/v1/chat/completions",
                    config.model,
                )
            else:
                server.hang_until_terminated(f"http://{config.master_ip}:{config.server_port}/health")

    server_returncode = server.proc.returncode
    # capfd captures inherited subprocess file descriptors; redirect_stdout would not.
    captured = capfd.readouterr()
    return _DeploymentResult(
        completions=completions,
        worker_log=captured.out + captured.err,
        is_master=config.is_master,
        is_decoder=config.disagg_cfg.is_decoder(config.cur_index) if config.disagg_cfg else False,
        server_was_running=server_was_running,
        server_returncode=server_returncode,
    )


@pytest.mark.asyncio
async def test_dual_path_matches_mooncake_layerwise_foundation(capfd: pytest.CaptureFixture[str]) -> None:
    capfd.readouterr()

    baseline = await _deploy_and_collect(BASELINE_CONFIG_PATH, capfd)
    await asyncio.sleep(BETWEEN_RUN_SETTLE_SECONDS)
    candidate = await _deploy_and_collect(CANDIDATE_CONFIG_PATH, capfd)

    assert baseline.is_master == candidate.is_master
    assert baseline.is_decoder == candidate.is_decoder

    # §12 bullets 2, 4, and 5: every node compares its own two runs with the
    # markers appropriate to its role; consumer-side evidence lives on the
    # decode node, not on the master.
    _assert_node_log_parity(baseline.worker_log, candidate.worker_log, is_decoder=baseline.is_decoder)

    # §12 bullet 3: candidate logs must not expose excluded stage-one behavior.
    _assert_markers_absent(candidate.worker_log, CANDIDATE_NEGATIVE_MARKERS, "candidate")

    # §12 bullet 6: both runs must remain healthy and terminate cleanly.
    # RemoteOpenAIServer teardown sends SIGTERM to the server process tree, so a
    # clean shutdown yields 0 or -SIGTERM; any other code is an abnormal exit.
    _assert_markers_absent(baseline.worker_log, FAILURE_MARKERS, "baseline")
    _assert_markers_absent(candidate.worker_log, FAILURE_MARKERS, "candidate")
    assert baseline.server_was_running
    assert candidate.server_was_running
    assert baseline.server_returncode in (0, -signal.SIGTERM)
    assert candidate.server_returncode in (0, -signal.SIGTERM)

    if not baseline.is_master:
        return

    # §12 bullet 1: every prompt must produce identical text and token counts.
    baseline_tokens = tuple(
        (record.prompt, record.content, record.completion_tokens) for record in baseline.completions
    )
    candidate_tokens = tuple(
        (record.prompt, record.content, record.completion_tokens) for record in candidate.completions
    )
    assert baseline_tokens == candidate_tokens

    # §12 bullet 2: request-terminal outcomes must match.
    assert tuple(record.finish_reason for record in baseline.completions) == tuple(
        record.finish_reason for record in candidate.completions
    )

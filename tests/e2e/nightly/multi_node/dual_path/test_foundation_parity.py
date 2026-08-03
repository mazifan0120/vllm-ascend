# SPDX-License-Identifier: Apache-2.0
"""PR-00 NPU foundation parity smoke test from the foundation specification §12.

This test requires the NPU multi-node nightly runner. The exact command, image,
model, topology, and result are recorded in ``.specs/dual-path-stage1/TRACKING.md``.
Run on that runner with:

``pytest -sv tests/e2e/nightly/multi_node/dual_path/test_foundation_parity.py``
"""

import asyncio
import signal
import time
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
PROXY_READY_TIMEOUT_SECONDS: Final = 300.0
PROXY_READY_POLL_SECONDS: Final = 1.0
BETWEEN_RUN_SETTLE_SECONDS: Final = 5.0

# The parent worker logs "Initializing Mooncake work" (sic) while the scheduler
# logs "Initializing Mooncake Scheduler"; both sides must be counted so the
# baseline/candidate initialization counts are comparable.
MOONCAKE_INIT_MARKERS: Final[tuple[str, ...]] = (
    "Initializing Mooncake Scheduler",
    "Initializing Mooncake work",
)
DUAL_PATH_INIT_MARKERS: Final[tuple[str, ...]] = (
    "Initializing DualPath Scheduler",
    "Initializing DualPath Worker",
)
TRANSFER_ENGINE_MARKERS: Final[tuple[str, ...]] = ("Transfer Engine",)
REGISTER_BUFFER_MARKERS: Final[tuple[str, ...]] = ("register_buffer",)
# Worker wording differs by connector version, so these are deliberately
# best-effort, case-insensitive evidence markers for a real remote prefill.
# "metaserver" is logged by the scheduler when it posts a remote-prefill
# request to the proxy metaserver (mooncake_layerwise_connector.py).
REMOTE_PREFILL_EVIDENCE_MARKERS: Final[tuple[str, ...]] = (
    "remote prefill",
    "remote-prefill",
    "metaserver",
    "mooncake transfer",
    "transfer engine",
    "register_buffer",
)
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
    mooncake_initializations: int
    dual_path_initializations: int
    transfer_engine: int
    register_buffer: int
    remote_prefill_evidence: int


@dataclass(frozen=True, slots=True)
class _DeploymentResult:
    completions: tuple[_CompletionRecord, ...]
    worker_log: str
    is_master: bool
    server_was_running: bool
    server_returncode: int | None


def _count_markers(normalized_text: str, markers: tuple[str, ...]) -> int:
    return sum(normalized_text.count(marker.casefold()) for marker in markers)


def _extract_marker_counts(text: str) -> _MarkerCounts:
    normalized_text = text.casefold()
    return _MarkerCounts(
        mooncake_initializations=_count_markers(normalized_text, MOONCAKE_INIT_MARKERS),
        dual_path_initializations=_count_markers(normalized_text, DUAL_PATH_INIT_MARKERS),
        transfer_engine=_count_markers(normalized_text, TRANSFER_ENGINE_MARKERS),
        register_buffer=_count_markers(normalized_text, REGISTER_BUFFER_MARKERS),
        remote_prefill_evidence=_count_markers(normalized_text, REMOTE_PREFILL_EVIDENCE_MARKERS),
    )


def _assert_worker_log_parity(baseline_log: str, candidate_log: str) -> None:
    baseline_counts = _extract_marker_counts(baseline_log)
    candidate_counts = _extract_marker_counts(candidate_log)

    assert baseline_counts.remote_prefill_evidence > 0, (
        "baseline logs do not show a remote prefill transfer; "
        f"searched for {REMOTE_PREFILL_EVIDENCE_MARKERS}"
    )
    assert candidate_counts.remote_prefill_evidence > 0, (
        "candidate logs do not show a remote prefill transfer; "
        f"searched for {REMOTE_PREFILL_EVIDENCE_MARKERS}"
    )
    assert baseline_counts.mooncake_initializations > 0
    assert candidate_counts.dual_path_initializations > 0
    assert baseline_counts.mooncake_initializations == candidate_counts.dual_path_initializations
    assert baseline_counts.transfer_engine == candidate_counts.transfer_engine
    assert baseline_counts.register_buffer == candidate_counts.register_buffer


def _assert_markers_absent(text: str, markers: tuple[str, ...], run_name: str) -> None:
    normalized_text = text.casefold()
    present_markers = tuple(marker for marker in markers if marker.casefold() in normalized_text)
    assert not present_markers, f"{run_name} logs contain forbidden markers: {present_markers}"


async def _wait_for_proxy_ready(health_url: str) -> None:
    deadline = time.monotonic() + PROXY_READY_TIMEOUT_SECONDS
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            try:
                response = await client.get(health_url)
            except httpx.RequestError:
                response = None
            if response is not None and response.is_success:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"proxy not ready after {PROXY_READY_TIMEOUT_SECONDS}s: {health_url}")
            await asyncio.sleep(PROXY_READY_POLL_SECONDS)


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
                await _wait_for_proxy_ready(f"http://{config.master_ip}:{proxy.proxy_port}/health")
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
        server_was_running=server_was_running,
        server_returncode=server_returncode,
    )


@pytest.mark.asyncio
async def test_dual_path_matches_mooncake_layerwise_foundation(capfd: pytest.CaptureFixture[str]) -> None:
    capfd.readouterr()

    baseline = await _deploy_and_collect(BASELINE_CONFIG_PATH, capfd)
    await asyncio.sleep(BETWEEN_RUN_SETTLE_SECONDS)
    candidate = await _deploy_and_collect(CANDIDATE_CONFIG_PATH, capfd)

    if not baseline.is_master:
        return

    assert candidate.is_master

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

    # §12 bullets 2, 4, and 5: worker markers must match and prove remote prefill.
    _assert_worker_log_parity(baseline.worker_log, candidate.worker_log)

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

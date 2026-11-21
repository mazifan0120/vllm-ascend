# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

LOAD_GET_HISTOGRAM_BUCKETS = (
    1e-3,
    5e-3,
    1e-2,
    5e-2,
    1e-1,
    2e-1,
    3e-1,
    4e-1,
    5e-1,
    7.5e-1,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
)
_PREFILL_PHASES = {"slot_acquire", "range_get", "range_put"}


@dataclass
class AscendStoreKVConnectorStats(KVConnectorStats):
    """Serializable AscendStore connector metrics."""

    def reset(self) -> None:
        self.data.clear()

    def is_empty(self) -> bool:
        return not self.data

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if durations := other.data.get("load_get_duration_seconds"):
            self.data.setdefault("load_get_duration_seconds", []).extend(durations)
        if num_keys := other.data.get("load_get_keys"):
            self.data["load_get_keys"] = self.data.get("load_get_keys", 0) + num_keys
        for phase, (calls, seconds) in other.data.get("prefill_phases", {}).items():
            self._add_prefill_phase(phase, calls, seconds)
        if "delayed_release_requests" in other.data:
            self.data["delayed_release_requests"] = other.data["delayed_release_requests"]
            self.data["delayed_release_blocks"] = other.data["delayed_release_blocks"]
        return self

    def reduce(self) -> dict[str, int | float]:
        reduced: dict[str, int | float] = {
            "ascend_store_delayed_release_requests": self.data.get("delayed_release_requests", 0),
            "ascend_store_delayed_release_blocks": self.data.get("delayed_release_blocks", 0),
        }
        if durations := self.data.get("load_get_duration_seconds"):
            reduced["ascend_store_load_get_count"] = len(durations)
            reduced["ascend_store_load_get_avg_ms"] = round(fmean(durations) * 1e3, 3)
            reduced["ascend_store_load_get_keys"] = self.data.get("load_get_keys", 0)
        for phase, (calls, seconds) in self.data.get("prefill_phases", {}).items():
            reduced[f"ascend_store_prefill_{phase}_count"] = calls
            reduced[f"ascend_store_prefill_{phase}_avg_ms"] = round(seconds * 1e3 / calls, 3)
        return reduced

    def record_operation(self, operation: str, duration_seconds: float, num_keys: int) -> None:
        if operation == "load_get":
            self.data.setdefault("load_get_duration_seconds", []).append(duration_seconds)
            self.data["load_get_keys"] = self.data.get("load_get_keys", 0) + num_keys
        elif operation in _PREFILL_PHASES:
            self._add_prefill_phase(operation, 1, duration_seconds)

    def _add_prefill_phase(self, phase: str, calls: int, seconds: float) -> None:
        phases = self.data.setdefault("prefill_phases", {})
        previous_calls, previous_seconds = phases.get(phase, (0, 0.0))
        phases[phase] = (previous_calls + calls, previous_seconds + seconds)

    def set_delayed_release(self, num_requests: int, num_blocks: int) -> None:
        self.data["delayed_release_requests"] = num_requests
        self.data["delayed_release_blocks"] = num_blocks


class AscendStorePromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._delayed_release_requests = create_metric_per_engine(
            self._gauge_cls(
                name="vllm:ascend_store_delayed_release_requests",
                documentation=(
                    "Number of finished requests whose KV cache block release is "
                    "delayed by an asynchronous AscendStore save."
                ),
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._delayed_release_blocks = create_metric_per_engine(
            self._gauge_cls(
                name="vllm:ascend_store_delayed_release_blocks",
                documentation=(
                    "Number of KV cache block references retained for finished "
                    "requests while asynchronous AscendStore saves complete."
                ),
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._load_get_duration = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:ascend_store_load_get_duration_seconds",
                documentation="Histogram of per-worker non-layerwise AscendStore Backend.get duration.",
                buckets=LOAD_GET_HISTOGRAM_BUCKETS,
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._load_get_keys = create_metric_per_engine(
            self._counter_cls(
                name="vllm:ascend_store_load_get_keys_total",
                documentation="Number of keys passed to completed AscendStore Backend.get calls.",
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._prefill_operations = self._counter_cls(
            name="vllm:ascend_store_prefill_operations_total",
            documentation="Number of completed P layerwise slot acquisitions or returned Store range calls.",
            labelnames=[*labelnames, "phase"],
        )
        self._prefill_seconds = self._counter_cls(
            name="vllm:ascend_store_prefill_operation_seconds_total",
            documentation="Total worker time in P layerwise slot acquisition or Store range calls.",
            labelnames=[*labelnames, "phase"],
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        metric = self._load_get_duration.get(engine_idx)
        if metric is not None:
            for duration in transfer_stats_data.get("load_get_duration_seconds", ()):
                metric.observe(duration)
        metric = self._load_get_keys.get(engine_idx)
        if metric is not None:
            metric.inc(transfer_stats_data.get("load_get_keys", 0))
        metric = self._delayed_release_requests.get(engine_idx)
        if metric is not None:
            if "delayed_release_requests" in transfer_stats_data:
                metric.set(transfer_stats_data["delayed_release_requests"])
        metric = self._delayed_release_blocks.get(engine_idx)
        if metric is not None:
            if "delayed_release_blocks" in transfer_stats_data:
                metric.set(transfer_stats_data["delayed_release_blocks"])
        for phase, (calls, seconds) in transfer_stats_data.get("prefill_phases", {}).items():
            labels = (*self.per_engine_labelvalues[engine_idx], phase)
            self._prefill_operations.labels(*labels).inc(calls)
            self._prefill_seconds.labels(*labels).inc(seconds)

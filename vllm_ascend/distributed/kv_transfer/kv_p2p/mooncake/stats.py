# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Transfer statistics and Prometheus metrics for Mooncake."""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine


@dataclass
class MooncakeKVConnectorStats(KVConnectorStats):
    """D2D transfer statistics collected by a Mooncake worker."""

    def __post_init__(self) -> None:
        if not self.data:
            self.reset()

    def reset(self) -> None:
        # Values must remain serializable because worker stats are aggregated
        # outside of the worker process.
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "bytes_transferred": [],
            "num_failed_transfers": [],
        }

    def record_transfer(self, transfer_duration: float, bytes_transferred: int) -> None:
        """Record one completed D2D transfer.

        ``transfer_duration`` is expressed in seconds.
        """
        self.data["transfer_duration"].append(transfer_duration)
        self.data["bytes_transferred"].append(bytes_transferred)

    def record_failed_transfer(self) -> None:
        self.data["num_failed_transfers"].append(1)

    def clone_and_reset(self) -> "MooncakeKVConnectorStats":
        previous = copy.copy(self)
        self.reset()
        return previous

    def is_empty(self) -> bool:
        return self.num_successful_transfers == 0 and not self.data["num_failed_transfers"]

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for key, values in other.data.items():
                accumulator = self.data[key]
                assert isinstance(accumulator, list)
                accumulator.extend(values)
        return self

    def reduce(self) -> dict[str, int | float]:
        if self.num_successful_transfers == 0:
            return {
                "Num successful transfers": 0,
                "Num failed transfers": len(self.data["num_failed_transfers"]),
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
            }

        durations = np.asarray(self.data["transfer_duration"])
        megabytes = np.asarray(self.data["bytes_transferred"]) / 2**20
        total_duration = durations.sum()
        throughput = megabytes.sum() / total_duration if total_duration > 0 else 0

        return {
            "Num successful transfers": self.num_successful_transfers,
            "Num failed transfers": len(self.data["num_failed_transfers"]),
            "Avg xfer time (ms)": round(durations.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(durations, 90).item() * 1e3, 3),
            "Avg MB per transfer": round(megabytes.mean(), 3),
            "Throughput (MB/s)": round(float(throughput), 3),
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class MooncakePromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for Mooncake D2D transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )

        transfer_duration = self._histogram_cls(
            name="vllm:mooncake_xfer_time_seconds",
            documentation=("Histogram of Mooncake KV cache D2D transfer duration."),
            buckets=[
                0.005,
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.2,
                0.3,
                0.5,
                0.75,
                1.0,
                5.0,
            ],
            labelnames=labelnames,
        )
        self.transfer_duration = create_metric_per_engine(transfer_duration, self.per_engine_labelvalues)

        bytes_transferred = self._histogram_cls(
            name="vllm:mooncake_bytes_transferred",
            documentation=("Histogram of bytes transferred per Mooncake KV cache D2D transfer."),
            buckets=[2 ** (10 + index) for index in range(1, 25, 2)],
            labelnames=labelnames,
        )
        self.bytes_transferred = create_metric_per_engine(bytes_transferred, self.per_engine_labelvalues)

        failed_transfers = self._counter_cls(
            name="vllm:mooncake_num_failed_transfers",
            documentation="Number of failed Mooncake KV cache D2D transfers.",
            labelnames=labelnames,
        )
        self.failed_transfers = create_metric_per_engine(failed_transfers, self.per_engine_labelvalues)

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        for duration in transfer_stats_data["transfer_duration"]:
            self.transfer_duration[engine_idx].observe(duration)
        for num_bytes in transfer_stats_data["bytes_transferred"]:
            self.bytes_transferred[engine_idx].observe(num_bytes)
        for failure in transfer_stats_data["num_failed_transfers"]:
            self.failed_transfers[engine_idx].inc(failure)


_LAYERWISE_WRITE_METRICS = {
    "batches": "Number of P-to-D layerwise native write submission attempts.",
    "waits": "Number of P-to-D layerwise native write completion waits.",
    "failures": "Number of failed P-to-D layerwise write submissions or waits.",
    "submit_seconds": "Total worker time in P-to-D layerwise native write submission.",
    "wait_seconds": "Total worker time waiting on P-to-D layerwise native writes.",
}
_LAYERWISE_DECODE_METRICS = {
    "prepare": "Number of layerwise D targets confirmed prepared by all ranks.",
    "prepare_seconds": "Total time from retaining D blocks to the last Prepare confirmation.",
    "ready": "Number of layerwise D targets confirmed ready by all ranks.",
    "ready_seconds": "Total time from D target publication to the last Ready confirmation.",
    "close_started": "Number of layerwise D targets for which the scheduler sent Close.",
    "close": "Number of layerwise D targets confirmed closed by all ranks.",
    "close_seconds": "Total time from sending Close to the last Closed confirmation.",
}


@dataclass
class LayerwiseTransferStats(KVConnectorStats):
    def record(self, submit: float, wait: float | None, success: bool) -> None:
        self.data["batches"] = self.data.get("batches", 0) + 1
        self.data["submit_seconds"] = self.data.get("submit_seconds", 0) + submit
        if wait is not None:
            self.data["waits"] = self.data.get("waits", 0) + 1
            self.data["wait_seconds"] = self.data.get("wait_seconds", 0) + wait
        if not success:
            self.data["failures"] = self.data.get("failures", 0) + 1

    def reset(self) -> None:
        self.data.clear()

    def is_empty(self) -> bool:
        return not self.data

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        for key, value in other.data.items():
            self.data[key] = self.data.get(key, 0) + value
        return self

    def reduce(self) -> dict[str, int | float]:
        return dict(self.data)


class LayerwiseTransferPromMetrics(KVConnectorPromMetrics):
    def __init__(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        roles = (
            (self._kv_transfer_config.is_kv_producer, "write", _LAYERWISE_WRITE_METRICS),
            (self._kv_transfer_config.is_kv_consumer, "decode", _LAYERWISE_DECODE_METRICS),
        )
        self._counters = {
            key: create_metric_per_engine(
                self._counter_cls(
                    name=f"vllm:mooncake_layerwise_{role}_{key}_total",
                    documentation=description,
                    labelnames=labelnames,
                ),
                per_engine_labelvalues,
            )
            for enabled, role, fields in roles
            if enabled
            for key, description in fields.items()
        }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        for key, value in transfer_stats_data.items():
            self._counters[key][engine_idx].inc(value)

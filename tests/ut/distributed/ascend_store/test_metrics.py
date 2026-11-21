# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from functools import partial
from unittest.mock import MagicMock

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metrics import (
    AscendStoreKVConnectorStats,
    AscendStorePromMetrics,
)


class _MetricChild:
    def __init__(self):
        self.value = 0
        self.observed: list[float] = []

    def set(self, value):
        self.value = value

    def inc(self, value=1):
        self.value += value

    def observe(self, value):
        self.observed.append(value)


class _Metric:
    def __init__(self, *args, **kwargs):
        self.child = _MetricChild()

    def labels(self, *args):
        return self.child


def test_stats_aggregate_and_reduce():
    first = AscendStoreKVConnectorStats()
    first.set_delayed_release(1, 2)
    first.record_operation("load_get", 0.01, 3)
    second = AscendStoreKVConnectorStats()
    second.set_delayed_release(2, 5)
    second.record_operation("load_get", 0.02, 5)

    first.aggregate(second)

    assert first.reduce() == {
        "ascend_store_delayed_release_requests": 2,
        "ascend_store_delayed_release_blocks": 5,
        "ascend_store_load_get_count": 2,
        "ascend_store_load_get_avg_ms": 15.0,
        "ascend_store_load_get_keys": 8,
    }


def test_prom_metrics_observe():
    metric_types = {Gauge: _Metric, Counter: _Metric, Histogram: _Metric}
    labelnames = ["model_name"]
    labelvalues = {0: ["test-model"]}
    ascend_store_prom = AscendStorePromMetrics(MagicMock(), metric_types, labelnames, labelvalues)
    ascend_store_prom.observe(
        {
            "delayed_release_requests": 2,
            "delayed_release_blocks": 5,
            "load_get_duration_seconds": [0.01, 0.02],
            "load_get_keys": 8,
        }
    )

    assert ascend_store_prom._delayed_release_requests[0].value == 2
    assert ascend_store_prom._delayed_release_blocks[0].value == 5
    assert ascend_store_prom._load_get_duration[0].observed == [0.01, 0.02]
    assert ascend_store_prom._load_get_keys[0].value == 8


def test_prefill_phase_counters_aggregate_and_label_results():
    registry = CollectorRegistry()
    metric_types = {cls: partial(cls, registry=registry) for cls in (Gauge, Counter, Histogram)}
    metrics = AscendStorePromMetrics(MagicMock(), metric_types, ["engine"], {0: ["0"]})
    counts = "vllm:ascend_store_prefill_operations_total"
    seconds = "vllm:ascend_store_prefill_operation_seconds_total"
    labels = {"engine": "0", "phase": "range_get"}
    assert registry.get_sample_value(counts, labels) is None

    first, second = AscendStoreKVConnectorStats(), AscendStoreKVConnectorStats()
    first.record_operation("range_get", 0.25, 10)
    second.record_operation("range_get", 0.5, 10)
    second.record_operation("slot_acquire", 0.125, 0)
    first.aggregate(second)
    metrics.observe(first.data)

    assert registry.get_sample_value(counts, labels) == 2
    assert registry.get_sample_value(seconds, labels) == 0.75
    assert registry.get_sample_value(seconds, {"engine": "0", "phase": "slot_acquire"}) == 0.125
    assert registry.get_sample_value(counts, {"engine": "0", "phase": "range_put"}) is None

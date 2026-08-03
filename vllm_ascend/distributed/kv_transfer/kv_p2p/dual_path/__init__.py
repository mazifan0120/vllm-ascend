# SPDX-License-Identifier: Apache-2.0
"""Dual-path KV-transfer connector package (PR-00 foundation).

This package provides ``DualPathConnector``, a behavior-preserving alias of
``MooncakeLayerwiseConnector`` for ordinary Layerwise workloads, together with
its frozen foundation configuration ``DualPathConfig``. PR-00 adds no DualPath
decision or data path; it only creates the safe Scheduler and Worker subclass
seams that later PRs build on.

Modules are imported lazily via the connector name registered in
``vllm_ascend.distributed.kv_transfer``; this ``__init__`` deliberately does not
eagerly import the connector (which pulls in mooncake / torch_npu).
"""

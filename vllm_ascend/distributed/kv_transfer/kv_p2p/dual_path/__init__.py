# SPDX-License-Identifier: Apache-2.0
"""Dual-path KV-transfer connector package.

This package provides ``DualPathConnector``, its configuration, path-decision
protocol, and route metadata. The connector extends Layerwise KV transfer with
Decode admission, Prefill-owned path selection, local Store-full handling, and
``PE_READ`` and ``DE_READ`` routes while preserving ordinary
Layerwise behavior.

Modules are imported lazily via the connector name registered in
``vllm_ascend.distributed.kv_transfer``; this ``__init__`` deliberately does not
eagerly import the connector (which pulls in mooncake / torch_npu).
"""

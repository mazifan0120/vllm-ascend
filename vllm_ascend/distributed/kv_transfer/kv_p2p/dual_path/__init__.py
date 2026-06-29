# SPDX-License-Identifier: Apache-2.0
"""Dual-path KV-transfer connector (Stage 1 foundation).

This package adds ``DualPathConnector``, a sibling sub-connector (alongside
``AscendStoreConnector``) for ``AscendMultiConnector`` that selects between
PE-Read and DE-Read KV load paths. Stage 1 ships the foundation:
``DualPathConfig`` (validated configuration) and the connector skeleton
inheriting ``MooncakeLayerwiseConnector``. The decision subsystem
(``PathStrategy`` / ``LinkMonitor`` / ``Topology``) and the PE/DE-Read execution
arrive in later stages.

Modules are imported lazily via the connector name registered in
``vllm_ascend.distributed.kv_transfer``; this ``__init__`` deliberately does not
eagerly import the connector (which pulls in mooncake / torch_npu).
"""

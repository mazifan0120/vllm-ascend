# SPDX-License-Identifier: Apache-2.0
"""Shared stubs for DualPath unit tests.

Mirrors the module-stub preamble of ``test_dual_path_connector.py``: the
optional Mooncake engine, torch_npu, and uvloop extensions are stubbed before
any vllm_ascend import so the suites run on a plain CPU checkout.
"""

import importlib.util
import sys
import types
from unittest.mock import MagicMock

import torch

_fake_engine = types.ModuleType("mooncake.engine")
_fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("mooncake.engine", _fake_engine)

_fake_torch_npu = types.ModuleType("torch_npu")
_fake_torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
_fake_torch_npu.npu = MagicMock()  # type: ignore[attr-defined]
_fake_torch_npu.npu.current_device = MagicMock(return_value=0)  # type: ignore[attr-defined]
_fake_torch_npu.npu.Stream = MagicMock  # type: ignore[attr-defined]
_fake_torch_npu.npu_fusion_attention = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("torch_npu", _fake_torch_npu)
torch.npu = _fake_torch_npu.npu  # type: ignore[attr-defined]

_fake_uvloop = types.ModuleType("uvloop")
_fake_uvloop.__spec__ = importlib.util.spec_from_loader("uvloop", loader=None)
sys.modules.setdefault("uvloop", _fake_uvloop)

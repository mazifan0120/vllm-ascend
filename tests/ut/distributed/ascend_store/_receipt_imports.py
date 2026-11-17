# SPDX-License-Identifier: Apache-2.0
"""Import pure receipt leaves without mocking an installed vLLM/backend."""

import importlib
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

_PROBE = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract"
_CHAIN = (
    ("vllm_ascend", _REPO_ROOT / "vllm_ascend"),
    ("vllm_ascend.distributed", _REPO_ROOT / "vllm_ascend" / "distributed"),
    ("vllm_ascend.distributed.kv_transfer", _REPO_ROOT / "vllm_ascend" / "distributed" / "kv_transfer"),
    (
        "vllm_ascend.distributed.kv_transfer.kv_p2p",
        _REPO_ROOT / "vllm_ascend" / "distributed" / "kv_transfer" / "kv_p2p",
    ),
)


def _ensure_package_chain() -> None:
    """Make the in-repo absolute imports work even when the real package
    __init__ files pull in torch/vllm that a CPU-only test box lacks: stub the
    intermediate packages with plain namespace-style modules so only the three
    pure-stdlib leaf modules load."""
    try:
        importlib.import_module(_PROBE)
        return
    except Exception:
        pass
    for name, path in _CHAIN:
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "__path__", None):
            continue
        stub = types.ModuleType(name)
        stub.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = stub
    importlib.invalidate_caches()


_ensure_package_chain()


# SPDX-License-Identifier: Apache-2.0
"""Configuration parsing and validation for ``DualPathConnector``.

``DualPathConfig`` carries the connector role and ``dual_path_control_port``.
KVPool and Mooncake settings are read directly from
``kv_connector_extra_config`` by their owning components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig

# Accepted connector extra-config keys. ``role`` and
# ``dual_path_control_port`` are stored on ``DualPathConfig``. KVPool keys are
# type-checked here; ``tls_config`` is only let through for its owning
# component, which reads it directly from ``kv_connector_extra_config``.
# Decode Store loads require ``load_async`` so synchronous Store I/O never
# runs in the Scheduler or model thread. ``consumer_is_to_put`` is unsupported:
# DualPath never puts into Store from the Decode side.
ALLOWED_EXTRA_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "role",
        "tls_config",
        "consumer_is_to_load",
        "load_async",
        "backend",
        "lookup_rpc_port",
        "mooncake_rpc_port",
        "discard_partial_chunks",
        "dual_path_control_port",
    }
)

# Values accepted for the connector ``role`` key.
SUPPORTED_ROLES: tuple[str, str] = ("prefill", "decode")

# Rendered once so every role-related error message lists the same choices.
_ROLE_CHOICES: str = " | ".join(repr(role) for role in SUPPORTED_ROLES)

MIN_TCP_PORT: int = 1
MAX_TCP_PORT: int = 65535


@dataclass(frozen=True)
class DualPathConfig:
    """Connector role and optional Decode control port."""

    role: Literal["prefill", "decode"]
    dual_path_control_port: int | None = None

    @classmethod
    def from_extra_config(
        cls, extra_config: dict[str, Any] | None, kv_transfer_config: KVTransferConfig
    ) -> DualPathConfig:
        """Build a validated connector config from the extra-config dict.

        Raises:
            ValueError: if any part of the extra config fails validation.
        """
        extra = extra_config or {}

        unknown = sorted(set(extra) - ALLOWED_EXTRA_CONFIG_KEYS)
        if unknown:
            msg = (
                f"DualPathConnector got unsupported kv_connector_extra_config "
                f"key(s) {unknown}; allowed keys: "
                f"{sorted(ALLOWED_EXTRA_CONFIG_KEYS)}."
            )
            raise ValueError(msg)

        role = extra.get("role")
        if role is None:
            raise ValueError(
                f"DualPathConnector requires 'role' in kv_connector_extra_config (choices: {_ROLE_CHOICES})."
            )
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"DualPathConnector role={role!r} is not supported; choices: {_ROLE_CHOICES}.")

        _validate_kvpool_passthrough(extra)
        _validate_decode_local_load_async(extra, role)
        _validate_role_capability(role, kv_transfer_config)
        dual_path_control_port = _validate_dual_path_control_port(extra, role)
        return cls(role=role, dual_path_control_port=dual_path_control_port)


def _validate_kvpool_passthrough(extra: dict[str, Any]) -> None:
    """Validate the KVPool pass-through keys.

    The values are consumed by the owned KVPool components from
    ``kv_connector_extra_config``; only type/shape validation happens here.
    """
    for key in ("consumer_is_to_load", "load_async", "discard_partial_chunks"):
        if key in extra and not isinstance(extra[key], bool):
            raise ValueError(f"DualPathConnector KVPool setting {key!r} must be a boolean; got {extra[key]!r}.")
    if "backend" in extra:
        backend = extra["backend"]
        if not isinstance(backend, str) or not backend:
            raise ValueError(f"DualPathConnector KVPool setting 'backend' must be a non-empty string; got {backend!r}.")
    for key in ("lookup_rpc_port", "mooncake_rpc_port"):
        if key in extra:
            port = extra[key]
            if isinstance(port, bool) or not isinstance(port, int) or port < 0:
                raise ValueError(
                    f"DualPathConnector KVPool setting {key!r} must be a non-negative integer; got {port!r}."
                )


def _validate_decode_local_load_async(extra: dict[str, Any], role: str) -> None:
    """Fail fast when a Decode local Store load would run synchronously."""
    if role == "decode" and extra.get("consumer_is_to_load") is True and extra.get("load_async") is not True:
        raise ValueError(
            "DualPathConnector role='decode' with 'consumer_is_to_load' True requires "
            "'load_async' True in kv_connector_extra_config; got "
            f"load_async={extra.get('load_async')!r} (synchronous Store I/O must not "
            "run in the Scheduler or model execution thread)."
        )


def _validate_role_capability(role: str, kv_transfer_config: KVTransferConfig) -> None:
    """Cross-check the declared connector role against the KV capability.

    A prefill Engine is the KV producer; a decode Engine is the KV consumer.
    ``kv_both`` satisfies either.
    """
    if role == "prefill" and not kv_transfer_config.is_kv_producer:
        raise ValueError(
            "DualPathConnector role='prefill' requires kv_role in "
            f"('kv_producer', 'kv_both'); got kv_role={kv_transfer_config.kv_role!r}."
        )
    if role == "decode" and not kv_transfer_config.is_kv_consumer:
        raise ValueError(
            "DualPathConnector role='decode' requires kv_role in "
            f"('kv_consumer', 'kv_both'); got kv_role={kv_transfer_config.kv_role!r}."
        )


def _validate_dual_path_control_port(extra: dict[str, Any], role: str) -> int | None:
    """Validate and return the dual-path control port.

    The port is required for role="decode" and validated here only as an
    integer (not bool) within 1..65535. It is rejected for role="prefill",
    which never consumes it.
    """
    port = extra.get("dual_path_control_port")
    if role == "decode" and port is None:
        raise ValueError(
            "DualPathConnector role='decode' requires 'dual_path_control_port' "
            "in kv_connector_extra_config (an integer between 1 and 65535)."
        )
    if role != "decode" and port is not None:
        raise ValueError(
            "DualPathConnector 'dual_path_control_port' is only consumed by role='decode'; "
            "remove it from the prefill-side configuration."
        )
    if port is None:
        return None
    if isinstance(port, bool) or not isinstance(port, int) or not MIN_TCP_PORT <= port <= MAX_TCP_PORT:
        raise ValueError(
            f"DualPathConnector 'dual_path_control_port' must be an integer between 1 and 65535; got {port!r}."
        )
    return port

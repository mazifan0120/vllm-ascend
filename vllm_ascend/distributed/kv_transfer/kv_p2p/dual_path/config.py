# SPDX-License-Identifier: Apache-2.0
"""Foundation configuration for ``DualPathConnector`` (PR-00).

PR-00 scope: foundation role parsing and fail-fast validation only. The only
DualPath-owned configuration field is ``role``; every other key the foundation
accepts (``tls_config``, ``prefill``, ``decode``) is inherited Mooncake
parallel/runtime configuration consumed by the parent implementation and is
passed through untouched.

Task-01 addition: the existing lookup-only KVPool settings
(``consumer_is_to_load``, ``backend``, ``lookup_rpc_port``,
``mooncake_rpc_port``, ``discard_partial_chunks``) are accepted as validated
pass-through keys. They are read directly from ``kv_connector_extra_config``
by the owned KVPool components, so they remain absent from the frozen
dataclass below.

All knobs are read from the connector's own ``kv_connector_extra_config``. No
environment variables are introduced (see AGENTS.md).

Fields that later PRs will own (path strategy, relay, path planner, monitor,
topology, shadow toggles) have no producer or consumer in PR-00 and therefore
fail fast instead of being parsed and silently ignored. If the parent Mooncake
connector later adds a required extra-config key, the compatibility guard test
fails until ``ALLOWED_EXTRA_CONFIG_KEYS`` and its parity tests are reviewed
together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig

# Extra-config keys the foundation accepts. ``role`` is the only DualPath-owned
# key; the rest are inherited Mooncake configuration consumed by the parent.
# Task-01 admits the existing lookup-only KVPool settings as pass-through keys:
# they are validated here but consumed directly from
# ``kv_connector_extra_config`` by the owned ``KVPoolScheduler``/``KVPoolWorker``,
# so they are NOT stored as ``DualPathConfig`` fields. ``load_async`` and
# ``consumer_is_to_put`` stay rejected: DualPath owns the Core-facing async
# admission result itself, and Decode-side put behavior is not part of Task-01.
# Task-03 adds ``dual_path_control_port``: required for role="decode", stored
# on ``DualPathConfig`` and consumed by the Task-03 control endpoint.
ALLOWED_EXTRA_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "role",
        "tls_config",
        "prefill",
        "decode",
        "consumer_is_to_load",
        "backend",
        "lookup_rpc_port",
        "mooncake_rpc_port",
        "discard_partial_chunks",
        "dual_path_control_port",
    }
)

# Keys promised by earlier drafts but removed from the foundation scope. They
# are rejected explicitly so a stale deployment config fails fast with a clear
# message instead of being treated as an unknown key.
REMOVED_FOUNDATION_FIELDS: frozenset[str] = frozenset(
    {
        "path_strategy",
        "relay",
        "path_planner",
        "monitor",
        "topology",
        "enable_value_function_shadow",
        "enable_link_monitor_shadow",
    }
)

# User-facing foundation roles. The legacy "pe"/"de" aliases are not accepted.
FOUNDATION_ROLES: tuple[str, str] = ("prefill", "decode")


@dataclass(frozen=True)
class DualPathConfig:
    """Parsed and validated PR-00 foundation configuration.

    The frozen dataclass carries no strategy, monitor, relay, planner,
    topology, Store, timeout, endpoint, or shadow field; those arrive only
    with the first later PR that has a real producer, consumer, and complete
    tests.
    """

    role: Literal["prefill", "decode"]
    dual_path_control_port: int | None = None

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any] | None, ktc: KVTransferConfig) -> DualPathConfig:
        """Build a validated foundation config from the extra-config dict.

        Raises:
            ValueError: on unknown keys (removed foundation fields called out
                explicitly), a missing/legacy/unknown role, or a
                role/KV-capability mismatch.
        """
        extra = extra_config or {}

        unknown = sorted(set(extra) - ALLOWED_EXTRA_CONFIG_KEYS)
        if unknown:
            msg = (
                f"DualPathConnector got unsupported kv_connector_extra_config "
                f"key(s) {unknown}; allowed keys: "
                f"{sorted(ALLOWED_EXTRA_CONFIG_KEYS)}."
            )
            removed = sorted(set(unknown) & REMOVED_FOUNDATION_FIELDS)
            if removed:
                msg += f" Key(s) {removed} were removed from the foundation scope and fail fast in PR-00."
            raise ValueError(msg)

        role = extra.get("role")
        if role is None:
            raise ValueError(
                "DualPathConnector requires 'role' in kv_connector_extra_config (choices: 'prefill' | 'decode')."
            )
        if role in ("pe", "de"):
            raise ValueError(
                f"DualPathConnector role={role!r} is a removed legacy alias; "
                "the foundation roles are exactly 'prefill' and 'decode'."
            )
        if role not in FOUNDATION_ROLES:
            raise ValueError(f"DualPathConnector role={role!r} is not supported; choices: 'prefill' | 'decode'.")

        _validate_kvpool_passthrough(extra)
        _validate_role_capability(role, ktc)
        dual_path_control_port = _validate_dual_path_control_port(extra, role)
        return cls(role=role, dual_path_control_port=dual_path_control_port)


def _validate_kvpool_passthrough(extra: dict[str, Any]) -> None:
    """Validate the Task-01 lookup-only KVPool pass-through keys.

    The values are consumed by the owned KVPool components from
    ``kv_connector_extra_config``; only type/shape validation happens here.
    """
    for key in ("consumer_is_to_load", "discard_partial_chunks"):
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


def _validate_role_capability(role: str, ktc: KVTransferConfig) -> None:
    """Cross-check the declared foundation role against the KV capability.

    A prefill Engine is the KV producer; a decode Engine is the KV consumer.
    ``kv_both`` satisfies either.
    """
    if role == "prefill" and not ktc.is_kv_producer:
        raise ValueError(
            "DualPathConnector role='prefill' requires kv_role in "
            f"('kv_producer', 'kv_both'); got kv_role={ktc.kv_role!r}."
        )
    if role == "decode" and not ktc.is_kv_consumer:
        raise ValueError(
            "DualPathConnector role='decode' requires kv_role in "
            f"('kv_consumer', 'kv_both'); got kv_role={ktc.kv_role!r}."
        )


def _validate_dual_path_control_port(extra: dict[str, Any], role: str) -> int | None:
    """Validate and return the Task-03 control port.

    The port is required for role="decode" and must be an integer (not bool)
    within 1..65535 so the DP-rank-derived endpoint stays in range. For
    role="prefill" it is optional, type/range-checked when present, and stored
    but not consumed by the prefill side.
    """
    port = extra.get("dual_path_control_port")
    if role == "decode" and port is None:
        raise ValueError(
            "DualPathConnector role='decode' requires 'dual_path_control_port' "
            "in kv_connector_extra_config (an integer between 1 and 65535)."
        )
    if port is None:
        return None
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError(
            f"DualPathConnector 'dual_path_control_port' must be an integer between 1 and 65535; got {port!r}."
        )
    return port

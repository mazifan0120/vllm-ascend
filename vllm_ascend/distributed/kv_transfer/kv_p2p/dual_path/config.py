# SPDX-License-Identifier: Apache-2.0
"""Configuration parsing and validation for ``DualPathConnector`` (Stage 1).

All knobs are read from the connector's own ``kv_connector_extra_config`` (the
free-form dict each sub-connector receives inside a ``MultiConnector``) using
``KVTransferConfig.get_from_extra_config``. No environment variables are
introduced: every other connector in this tree is config-driven, and adding
``VLLM_ASCEND_*`` envs would diverge from that pattern (see AGENTS.md).

Stage 1 scope: only the subset of options needed by the *foundation* is
enforced here. The PE-Read / DE-Read execution and the full RelayService
protocol are not wired yet, so their knobs are parsed but a few are hard-rejected
(``path_planner.enabled``, unsupported ``relay.control_plane``) to fail fast
rather than silently no-op.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig

# The six value-function feature names (doc "DualPath 内部" / PathStrategy).
# Kept in sync with the PathStrategy value function; validated here so a typo fails fast.
PATH_STRATEGY_FEATURES: tuple[str, ...] = (
    "f_pe_io_headroom",
    "f_de_io_headroom",
    "f_net_headroom",
    "f_link_latency",
    "f_store_hit",
    "f_failure_penalty",
)

# Default value-function weights: main = link pressure / congestion
# (io/net/latency), minor = hit correction + failure penalty. Mirrors the doc's
# "配置项一览".
DEFAULT_WEIGHTS: dict[str, float] = {
    "f_pe_io_headroom": 1.2,
    "f_de_io_headroom": 1.2,
    "f_net_headroom": 1.2,
    "f_link_latency": 1.0,
    "f_store_hit": 0.3,
    "f_failure_penalty": 1.5,
}

# Default value-function params (consumed by the PathStrategy value function;
# documented here as the source of truth for the defaults).
DEFAULT_PARAMS: dict[str, Any] = {
    "mu_rtt": 30,  # us, RTT Gaussian center for f_link_latency
    "sigma_rtt": 10,  # us, RTT Gaussian spread for f_link_latency
    "L_max": 50.0,  # ms, IO-pressure latency ceiling (headroom proxy numerator)
    "N_fail": 5,  # rolling-window size for f_failure_penalty
}

DEFAULT_DECISION_WINDOW_MS: int = 50

DEFAULT_MONITOR_WINDOW_MS: int = 200
DEFAULT_MONITOR_EWMA_ALPHA: float = 0.3
DEFAULT_MONITOR_SOURCES: dict[str, dict[str, Any]] = {
    "mooncake": {"enabled": True, "period_ms": 100},
    "nic_sysfs": {"enabled": True, "period_ms": 200},
    "prometheus": {"enabled": False, "url": "http://localhost:9090"},
    "store_stats": {"enabled": True, "period_ms": 500},
}

DEFAULT_PATH_PLANNER_K_PATHS: int = 4
DEFAULT_PATH_PLANNER_COST_WEIGHTS: dict[str, float] = {
    "alpha": 1.0,
    "beta": 0.2,
    "gamma": 1.5,
    "delta": 5.0,
    "epsilon": 0.1,
}

DEFAULT_RELAY_ZMQ: dict[str, Any] = {"rep_port": 5555, "pub_port": 5556}

DEFAULT_TOPOLOGY_SOURCE_PRIORITY: tuple[str, ...] = ("ubutils", "sysfs", "udev", "static")

# Stage 1 only supports the ZMQ control plane (it reuses the layerwise ZMQ side
# channel). The doc's `zmq | urpc` choice defers `urpc` to Stage 2.
SUPPORTED_CONTROL_PLANES: tuple[str, ...] = ("zmq",)
SUPPORTED_PATH_STRATEGY_TYPES: tuple[str, ...] = ("static", "value_function", "adaptive")


@dataclass
class ValueFunctionCfg:
    """Weights / params / window for the PathStrategy value function."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    decision_window_ms: int = DEFAULT_DECISION_WINDOW_MS


@dataclass
class PathStrategyCfg:
    type: str = "value_function"  # static | value_function | adaptive
    value_function: ValueFunctionCfg = field(default_factory=ValueFunctionCfg)


@dataclass
class RelayCfg:
    control_plane: str = "zmq"
    zmq: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_RELAY_ZMQ))
    urpc: dict[str, Any] = field(default_factory=dict)


@dataclass
class PathPlannerCfg:
    # Stage 1: always disabled (multi-path slicing arrives in Stage 2 cascade).
    enabled: bool = False
    k_paths: int = DEFAULT_PATH_PLANNER_K_PATHS
    cost_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PATH_PLANNER_COST_WEIGHTS)
    )


@dataclass
class MonitorCfg:
    window_ms: int = DEFAULT_MONITOR_WINDOW_MS
    ewma_alpha: float = DEFAULT_MONITOR_EWMA_ALPHA
    sources: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_MONITOR_SOURCES.items()}
    )


@dataclass
class TopologyCfg:
    source_priority: tuple[str, ...] = DEFAULT_TOPOLOGY_SOURCE_PRIORITY
    static_yaml: str | None = None


@dataclass
class DualPathConfig:
    """Parsed + validated Stage-1 configuration for DualPathConnector."""

    role: Literal["pe", "de"]
    path_strategy: PathStrategyCfg
    relay: RelayCfg
    path_planner: PathPlannerCfg
    monitor: MonitorCfg
    topology: TopologyCfg

    @classmethod
    def from_extra_config(
        cls, extra_config: dict[str, Any] | None, ktc: "KVTransferConfig"
    ) -> "DualPathConfig":
        """Build a validated config from the connector's extra_config dict."""
        extra = extra_config or {}

        role = extra.get("role")
        if role not in ("pe", "de"):
            raise ValueError(
                "DualPathConnector requires 'role' in kv_connector_extra_config "
                "(choices: 'pe' | 'de')."
            )
        _validate_role(role, ktc)

        path_strategy = _parse_path_strategy(extra.get("path_strategy") or {})
        relay = _parse_relay(extra.get("relay") or {})
        path_planner = _parse_path_planner(extra.get("path_planner") or {})
        monitor = _parse_monitor(extra.get("monitor") or {})
        topology = _parse_topology(extra.get("topology") or {})

        # Stage 1 invariants.
        if path_planner.enabled:
            raise NotImplementedError(
                "DualPathConnector path_planner.enabled=True is not supported in "
                "Stage 1 (multi-path slicing arrives in Stage 2 cascade)."
            )
        if relay.control_plane not in SUPPORTED_CONTROL_PLANES:
            raise NotImplementedError(
                f"DualPathConnector relay.control_plane={relay.control_plane!r} is "
                f"not supported in Stage 1; supported: {SUPPORTED_CONTROL_PLANES}."
            )
        if path_strategy.type not in SUPPORTED_PATH_STRATEGY_TYPES:
            raise ValueError(
                f"DualPathConnector path_strategy.type={path_strategy.type!r} is "
                f"not supported; choices: {SUPPORTED_PATH_STRATEGY_TYPES}."
            )

        return cls(
            role=role,
            path_strategy=path_strategy,
            relay=relay,
            path_planner=path_planner,
            monitor=monitor,
            topology=topology,
        )


def _validate_role(role: str, ktc: "KVTransferConfig") -> None:
    """Cross-check the declared connector role against kv_role.

    PE (prefill engine) corresponds to the KV producer; DE (decode engine) to the
    KV consumer. ``kv_both`` satisfies either.
    """
    if role == "pe" and not ktc.is_kv_producer:
        raise ValueError(
            "DualPathConnector role='pe' requires kv_role in ('kv_producer', "
            f"'kv_both'); got kv_role={ktc.kv_role!r}."
        )
    if role == "de" and not ktc.is_kv_consumer:
        raise ValueError(
            "DualPathConnector role='de' requires kv_role in ('kv_consumer', "
            f"'kv_both'); got kv_role={ktc.kv_role!r}."
        )


def _parse_path_strategy(raw: dict[str, Any]) -> PathStrategyCfg:
    ps_type = raw.get("type", "value_function")
    vf_raw = raw.get("value_function") or {}
    weights = dict(DEFAULT_WEIGHTS)
    weights.update(
        {k: float(v) for k, v in (vf_raw.get("weights") or {}).items()}
    )
    _validate_weights(weights)
    params = dict(DEFAULT_PARAMS)
    params.update({k: v for k, v in (vf_raw.get("params") or {}).items()})
    decision_window_ms = int(vf_raw.get("decision_window_ms", DEFAULT_DECISION_WINDOW_MS))
    if decision_window_ms <= 0:
        raise ValueError(
            "DualPathConnector path_strategy.value_function.decision_window_ms "
            "must be > 0."
        )
    return PathStrategyCfg(
        type=ps_type,
        value_function=ValueFunctionCfg(
            weights=weights, params=params, decision_window_ms=decision_window_ms
        ),
    )


def _validate_weights(weights: dict[str, float]) -> None:
    unknown = set(weights) - set(PATH_STRATEGY_FEATURES)
    if unknown:
        raise ValueError(
            f"Unknown path_strategy weights: {sorted(unknown)}. "
            f"Known features: {list(PATH_STRATEGY_FEATURES)}."
        )
    if all(float(w) == 0.0 for w in weights.values()):
        logger.warning(
            "All DualPathConnector path_strategy weights are zero; decisions will "
            "be degenerate (always pe_read)."
        )


def _parse_relay(raw: dict[str, Any]) -> RelayCfg:
    zmq_cfg = dict(DEFAULT_RELAY_ZMQ)
    zmq_cfg.update(raw.get("zmq") or {})
    return RelayCfg(
        control_plane=raw.get("control_plane", "zmq"),
        zmq=zmq_cfg,
        urpc=dict(raw.get("urpc") or {}),
    )


def _parse_path_planner(raw: dict[str, Any]) -> PathPlannerCfg:
    cost_weights = dict(DEFAULT_PATH_PLANNER_COST_WEIGHTS)
    cost_weights.update(
        {k: float(v) for k, v in (raw.get("cost_weights") or {}).items()}
    )
    return PathPlannerCfg(
        enabled=bool(raw.get("enabled", False)),
        k_paths=int(raw.get("k_paths", DEFAULT_PATH_PLANNER_K_PATHS)),
        cost_weights=cost_weights,
    )


def _parse_monitor(raw: dict[str, Any]) -> MonitorCfg:
    sources = {k: dict(v) for k, v in DEFAULT_MONITOR_SOURCES.items()}
    for name, overrides in (raw.get("sources") or {}).items():
        sources.setdefault(name, {}).update(overrides or {})
    return MonitorCfg(
        window_ms=int(raw.get("window_ms", DEFAULT_MONITOR_WINDOW_MS)),
        ewma_alpha=float(raw.get("ewma_alpha", DEFAULT_MONITOR_EWMA_ALPHA)),
        sources=sources,
    )


def _parse_topology(raw: dict[str, Any]) -> TopologyCfg:
    priority = tuple(
        raw.get("source_priority", DEFAULT_TOPOLOGY_SOURCE_PRIORITY)
    )
    static_yaml = raw.get("static_yaml")
    if static_yaml is not None:
        # Minimal Stage-1 check: the file must exist. Full two-layer schema +
        # sufficiency validation lands later via StaticFileTopologySource,
        # but we fail fast here so a broken path is caught at deployment time
        # rather than at first request.
        if not os.path.isfile(str(static_yaml)):
            raise FileNotFoundError(
                f"DualPathConnector topology.static_yaml={static_yaml!r} does not "
                "exist; provide a valid topology file or unset it."
            )
        logger.info(
            "DualPathConnector topology.static_yaml=%s configured; full schema "
            "validation runs later.",
            static_yaml,
        )
    return TopologyCfg(source_priority=priority, static_yaml=static_yaml)

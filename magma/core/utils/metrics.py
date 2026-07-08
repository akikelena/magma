"""Prometheus metrics export, wired at the application boundary
(`magma.orchestration.MagmaWorkflow`), not into individual agents, to
keep agent modules dependency-light and independently testable without
a metrics backend installed.

`prometheus_client` is imported lazily: importing this module, and even
using `MagmaMetrics` when the package isn't installed, both work -- a
`_NullMetrics` stand-in silently no-ops every call instead. Metrics only
require the real dependency once `start_metrics_server` is actually
called with `prometheus_client` present, mirroring the lazy-import
pattern used for every other optional backend in this repo (neo4j,
weaviate, xgboost, sklearn, spacy, anthropic, crewai).
"""
from __future__ import annotations

import abc


class _Metrics(abc.ABC):
    @abc.abstractmethod
    def record_episode(self, status: str) -> None:
        """Increment the episode counter for a completed workflow run,
        labeled by its terminal `WorkflowStatus` value."""

    @abc.abstractmethod
    def record_validation_outcome(self, passed: bool) -> None:
        """Increment the validation-outcome counter (Automated Validation
        Agent's per-artifact pass/fail result)."""

    @abc.abstractmethod
    def observe_phase_duration(self, phase: str, seconds: float) -> None:
        """Record how long one top-level phase (phase1/phase2/phase3/
        phase4) took for one episode, into a duration histogram."""

    @abc.abstractmethod
    def record_escalation(self, reason: str) -> None:
        """Increment the escalation counter each time the Human Expert
        Agent's queue grows, labeled by `EscalationReason`. Pending-review
        backlog is an operational signal (a reviewer isn't keeping up, or
        a bad confidence threshold is escalating too much) that belongs
        next to the other workflow-outcome metrics."""


class _NullMetrics(_Metrics):
    """No-op implementation used when `prometheus_client` isn't
    installed, so `MagmaWorkflow` can unconditionally record metrics
    without every caller needing to check availability first."""

    def record_episode(self, status: str) -> None:
        return None

    def record_validation_outcome(self, passed: bool) -> None:
        return None

    def observe_phase_duration(self, phase: str, seconds: float) -> None:
        return None

    def record_escalation(self, reason: str) -> None:
        return None


class PrometheusMetrics(_Metrics):
    """Real Prometheus-backed metrics. Constructing this requires
    `prometheus_client` to be installed; use `get_metrics()` rather than
    instantiating this directly so callers automatically fall back to
    `_NullMetrics` when it isn't."""

    def __init__(self) -> None:
        import prometheus_client

        self._episodes_total = prometheus_client.Counter(
            "magma_episodes_total", "MAGMA workflow episodes by terminal status", ["status"]
        )
        self._validation_outcomes_total = prometheus_client.Counter(
            "magma_validation_outcomes_total", "Automated Validation Agent per-artifact outcomes", ["outcome"]
        )
        self._phase_duration_seconds = prometheus_client.Histogram(
            "magma_phase_duration_seconds", "Wall-clock duration of each top-level MAGMA phase", ["phase"]
        )
        self._escalations_total = prometheus_client.Counter(
            "magma_escalations_total", "Human Expert Agent escalations by EscalationReason", ["reason"]
        )

    def record_episode(self, status: str) -> None:
        self._episodes_total.labels(status=status).inc()

    def record_validation_outcome(self, passed: bool) -> None:
        self._validation_outcomes_total.labels(outcome="pass" if passed else "fail").inc()

    def observe_phase_duration(self, phase: str, seconds: float) -> None:
        self._phase_duration_seconds.labels(phase=phase).observe(seconds)

    def record_escalation(self, reason: str) -> None:
        self._escalations_total.labels(reason=reason).inc()


_singleton: _Metrics | None = None


def get_metrics() -> _Metrics:
    """Returns a process-wide `PrometheusMetrics` if `prometheus_client`
    is installed, otherwise a silently-no-op `_NullMetrics` -- so
    `MagmaWorkflow(metrics=get_metrics())` is always safe to call.

    Cached as a singleton rather than constructed fresh each call:
    `prometheus_client` registers each Counter/Histogram against one
    process-global registry, so a second `PrometheusMetrics()` in the
    same process (e.g. a second `MagmaWorkflow` instance) would raise
    "Duplicated timeseries in CollectorRegistry" instead of returning a
    usable object. Metrics are inherently process-global, so one shared
    instance is also the semantically correct choice, not just a
    workaround."""
    global _singleton
    if _singleton is not None:
        return _singleton
    try:
        _singleton = PrometheusMetrics()
    except ImportError:
        _singleton = _NullMetrics()
    return _singleton


def start_metrics_server(port: int) -> bool:
    """Starts Prometheus's built-in HTTP server exposing /metrics on
    `port` (the application boundary named in the observability
    docstring; `PROMETHEUS_PORT` in .env.example). Returns True if the
    server was started, False if `prometheus_client` isn't installed (in
    which case the caller should log a warning and continue -- metrics
    export is an operational nicety, not a correctness requirement)."""
    try:
        import prometheus_client
    except ImportError:
        return False
    prometheus_client.start_http_server(port)
    return True

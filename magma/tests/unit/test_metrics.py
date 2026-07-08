"""Unit tests for Prometheus metrics export.

Runs against whichever backend is actually importable in this
environment: if prometheus_client is installed, exercises the real
PrometheusMetrics path (catching real bugs like registry collisions);
otherwise falls back to asserting the NullMetrics no-op contract. Either
way, get_metrics() must never raise.
"""
from __future__ import annotations

import pytest

from magma.core.utils import metrics as metrics_module


def test_get_metrics_never_raises_and_is_a_singleton():
    m1 = metrics_module.get_metrics()
    m2 = metrics_module.get_metrics()
    assert m1 is m2


def test_get_metrics_methods_are_safe_to_call_repeatedly():
    m = metrics_module.get_metrics()
    # Must not raise, regardless of backend, and must be callable many
    # times (this is what MagmaWorkflow does across many episodes).
    for _ in range(3):
        m.record_episode("completed")
        m.record_validation_outcome(True)
        m.record_validation_outcome(False)
        m.observe_phase_duration("phase1", 0.01)
        m.record_escalation("low_confidence")


def test_second_prometheus_metrics_instance_does_not_collide():
    """Locks in the singleton fix: prometheus_client registers each
    Counter/Histogram against one process-global registry, so a second
    independently-constructed PrometheusMetrics() used to raise
    'Duplicated timeseries in CollectorRegistry'. Two MagmaWorkflow
    instances in the same process (routine in this test suite) must not
    hit that."""
    pytest.importorskip("prometheus_client")
    # Ensure the process-wide singleton exists (idempotent either way --
    # some other test in this session may already have created it).
    metrics_module.get_metrics()
    # A direct, uncached instantiation collides with whatever already
    # registered those metric names in the process-global registry --
    # exactly why get_metrics() is the supported entry point, not
    # PrometheusMetrics() directly.
    with pytest.raises(ValueError):
        metrics_module.PrometheusMetrics()
    # get_metrics() itself -- what MagmaWorkflow actually calls -- must
    # never raise, no matter how many times it's called.
    via_get_metrics_1 = metrics_module.get_metrics()
    via_get_metrics_2 = metrics_module.get_metrics()
    assert via_get_metrics_1 is via_get_metrics_2


def test_null_metrics_no_ops_without_error():
    null = metrics_module._NullMetrics()
    assert null.record_episode("completed") is None
    assert null.record_validation_outcome(True) is None
    assert null.observe_phase_duration("phase1", 1.0) is None
    assert null.record_escalation("low_confidence") is None


def test_start_metrics_server_returns_false_without_prometheus_client(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "prometheus_client":
            raise ImportError("simulated: prometheus_client not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    assert metrics_module.start_metrics_server(9099) is False

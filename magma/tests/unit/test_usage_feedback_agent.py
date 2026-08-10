import pytest

from magma.core.governance.audit_log import AuditLogger
from magma.phase4_improvement import UsageFeedbackAgent
from magma.phase4_improvement.usage_feedback_agent import SuccessProbabilityEstimator


def test_analyze_counts_actions():
    audit = AuditLogger()
    audit.log("Documentation Processing Agent", "process_source", episode_id="ep1")
    audit.log("Documentation Processing Agent", "process_source", episode_id="ep2")
    audit.log("Change Detection Agent", "detect_changes", episode_id="ep1")
    agent = UsageFeedbackAgent()
    result = agent.process(audit.all())
    assert ("process_source", 2) in result.output.slowest_actions


def test_success_estimator_heuristic_fallback_without_fit():
    estimator = SuccessProbabilityEstimator()
    high = estimator.predict_proba([0.9, 0.9, 0.9])
    low = estimator.predict_proba([0.1, 0.1, 0.1])
    assert high > 0.5 > low


def test_gradient_boosted_threshold():
    estimator = SuccessProbabilityEstimator()
    assert not estimator.should_use_gradient_boosted(50)
    assert estimator.should_use_gradient_boosted(500)


def test_slowest_actions_by_duration_uses_real_elapsed_time():
    """Locks in the fix: duration must come from actual timed LLM calls,
    not be inferred from call counts."""
    import json
    import time

    from magma.llm_providers import MockLLMProvider
    from magma.phase1_documentation_detection.change_detection_agent import (
        ChangeDetectionAgent,
        ChangeDetectionRequest,
        FragmentVersionPair,
    )

    def slow_responder(prompt: str) -> str:
        time.sleep(0.005)
        return json.dumps({
            "change_type": "MINOR", "confidence": 0.9, "intent_shift": 0.2,
            "compatibility_preserved": True, "affected_entities": [], "rationale": "x",
        })

    audit = AuditLogger()
    agent = ChangeDetectionAgent(llm_provider=MockLLMProvider(responder=slow_responder), audit_logger=audit)
    agent.process(ChangeDetectionRequest("V", "1", "2", [FragmentVersionPair("f1", "a\n", "b\n")]))

    report = UsageFeedbackAgent().analyze(audit.all())
    assert report.slowest_actions_by_duration_ms
    action, duration_ms = report.slowest_actions_by_duration_ms[0]
    assert action == "detect_changes"
    assert duration_ms >= 5.0


def test_success_estimator_fits_a_real_model_at_or_above_minimum_examples():
    """Locks in the real scikit-learn LogisticRegression path (below-5
    heuristic fallback is covered by test_success_estimator_heuristic_
    fallback_without_fit above). Skipped if scikit-learn isn't
    installed -- this repo's CI doesn't require it."""
    pytest.importorskip("sklearn")
    estimator = SuccessProbabilityEstimator()
    features = [[0.9, 0.9, 0.9], [0.8, 0.9, 0.8], [0.9, 0.8, 0.9], [0.1, 0.1, 0.1], [0.2, 0.1, 0.2]]
    labels = [1, 1, 1, 0, 0]

    estimator.fit(features, labels)

    assert estimator._fitted
    assert estimator._model is not None
    assert estimator.predict_proba([0.95, 0.95, 0.95]) > 0.5
    assert estimator.predict_proba([0.05, 0.05, 0.05]) < 0.5

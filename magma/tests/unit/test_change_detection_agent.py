import json

from magma.llm_providers import MockLLMProvider
from magma.phase1_documentation_detection.change_detection_agent import (
    ChangeDetectionAgent,
    ChangeDetectionRequest,
    FragmentVersionPair,
)


def _responder(payload):
    return lambda prompt: json.dumps(payload)


def test_identical_fragment_is_skipped():
    agent = ChangeDetectionAgent(llm_provider=MockLLMProvider())
    result = agent.process(ChangeDetectionRequest("V", "1", "2", [FragmentVersionPair("f1", "same\n", "same\n")]))
    assert result.output.changes == []


def test_editorial_only_change_is_filtered_before_llm():
    llm = MockLLMProvider()
    agent = ChangeDetectionAgent(llm_provider=llm)
    result = agent.process(ChangeDetectionRequest("V", "1", "2", [FragmentVersionPair("f1", "a\n\n", "a\n")]))
    assert len(result.output.changes) == 1
    assert result.output.changes[0].change_type.name == "EDITORIAL"
    assert llm.calls == []  # LLM stage must never be invoked for editorial-only diffs


def test_breaking_change_classified_from_llm_response():
    payload = {
        "change_type": "BREAKING", "confidence": 0.9, "intent_shift": 0.8,
        "compatibility_preserved": False, "affected_entities": ["x"], "rationale": "removed",
    }
    llm = MockLLMProvider(responder=_responder(payload))
    agent = ChangeDetectionAgent(llm_provider=llm)
    result = agent.process(ChangeDetectionRequest("V", "1", "2", [FragmentVersionPair("f1", "a\n", "b\n")]))
    assert result.output.changes[0].change_type.name == "BREAKING"
    assert not result.escalate


def test_low_confidence_flagged_as_edge_case_and_escalates():
    payload = {
        "change_type": "MAJOR", "confidence": 0.3, "intent_shift": 0.3,
        "compatibility_preserved": True, "affected_entities": [], "rationale": "ambiguous",
    }
    llm = MockLLMProvider(responder=_responder(payload))
    agent = ChangeDetectionAgent(llm_provider=llm, confidence_threshold=0.7)
    result = agent.process(ChangeDetectionRequest("V", "1", "2", [FragmentVersionPair("f1", "a\n", "b\n")]))
    assert result.output.changes[0].is_edge_case
    assert result.escalate


def test_unparseable_llm_output_defaults_to_edge_case_and_escalates():
    """Regression test: unparseable output used to set is_edge_case=True
    and bump edge_case_count (so result.escalate was True) without ever
    setting escalation_reason on the DetectedChange, so it was silently
    excluded from change_reasons and Phase1Result.requires_human_review
    stayed False -- the exact case that most needs review (confidence 0.0)
    never reached the Human Expert Agent."""
    llm = MockLLMProvider(responder=lambda p: "not json at all")
    agent = ChangeDetectionAgent(llm_provider=llm)
    result = agent.process(ChangeDetectionRequest("V", "1", "2", [FragmentVersionPair("f1", "a\n", "b\n")]))
    assert result.output.changes[0].is_edge_case
    assert result.output.changes[0].escalation_reason == "low_confidence"
    assert result.escalate
    assert result.escalation_reason == "low_confidence"


def test_security_sensitive_content_escalates_even_with_high_confidence():
    """Locks in the fix: the security-critical trigger must fire from the
    actual diff/rationale content, independent of classifier confidence."""
    payload = {
        "change_type": "MINOR", "confidence": 0.95, "intent_shift": 0.2,
        "compatibility_preserved": True, "affected_entities": [], "rationale": "wording only",
    }
    llm = MockLLMProvider(responder=_responder(payload))
    agent = ChangeDetectionAgent(llm_provider=llm)
    result = agent.process(ChangeDetectionRequest(
        "V", "1", "2", [FragmentVersionPair("f1", "use basic auth", "use OAuth2 authentication with token refresh")]
    ))
    assert result.escalate
    assert result.escalation_reason == "security_critical"
    assert result.output.changes[0].escalation_reason == "security_critical"


def test_non_security_low_confidence_still_reports_low_confidence_reason():
    payload = {
        "change_type": "MAJOR", "confidence": 0.3, "intent_shift": 0.3,
        "compatibility_preserved": True, "affected_entities": [], "rationale": "ambiguous wording",
    }
    llm = MockLLMProvider(responder=_responder(payload))
    agent = ChangeDetectionAgent(llm_provider=llm, confidence_threshold=0.7)
    result = agent.process(ChangeDetectionRequest("V", "1", "2", [FragmentVersionPair("f1", "a\n", "b\n")]))
    assert result.escalation_reason == "low_confidence"

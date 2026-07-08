from magma.core.governance.audit_log import AuditLogger
from magma.llm_providers import MockLLMProvider
from magma.phase4_improvement import ExplainabilityAgent
from magma.phase4_improvement.explainability_agent import ExplanationRequest


def test_produces_narrative_when_records_exist():
    audit = AuditLogger()
    audit.log("Change Detection Agent", "detect_changes", episode_id="ep1", change_count=1)
    agent = ExplainabilityAgent(llm_provider=MockLLMProvider(responder=lambda p: "concise rationale"))
    result = agent.process(ExplanationRequest(episode_id="ep1", audit_records=audit.for_episode("ep1"), validation_passed=True))
    assert result.success
    assert result.output.is_complete
    assert result.output.narrative == "concise rationale"
    assert result.output.confidence == 0.9


def test_incomplete_when_no_records():
    agent = ExplainabilityAgent(llm_provider=MockLLMProvider())
    result = agent.process(ExplanationRequest(episode_id="ep1", audit_records=[]))
    assert not result.success
    assert not result.output.is_complete

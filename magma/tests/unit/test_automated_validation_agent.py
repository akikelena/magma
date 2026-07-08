from magma.core.models.execution import GeneratedArtifact
from magma.phase3_validation import AutomatedValidationAgent, AutomatedValidationRequest


def _artifact(content="def f():\n    return 1", tests="def test_f():\n    assert f() == 1"):
    return GeneratedArtifact(template_id="t1", content=content, test_cases=tests)


def test_all_layers_pass_for_clean_artifact():
    agent = AutomatedValidationAgent()
    result = agent.process(AutomatedValidationRequest(artifact=_artifact()))
    assert result.success
    assert len(result.output.layer_results) == 6
    assert not result.escalate


def test_syntax_failure_short_circuits_before_later_layers():
    agent = AutomatedValidationAgent()
    result = agent.process(AutomatedValidationRequest(artifact=_artifact(content="def f(:")))
    assert not result.success
    assert len(result.output.layer_results) == 1
    assert result.output.layer_results[0].layer.name == "SYNTAX"


def test_missing_tests_fails_functional_layer():
    agent = AutomatedValidationAgent()
    result = agent.process(AutomatedValidationRequest(artifact=_artifact(tests="")))
    assert not result.success
    assert result.output.first_failure.layer.name == "FUNCTIONAL"


def test_security_pattern_fails_security_layer():
    agent = AutomatedValidationAgent()
    # eval('1') keeps the artifact runnable (so it legitimately clears the
    # earlier FUNCTIONAL layer) while still tripping the security scanner.
    result = agent.process(AutomatedValidationRequest(
        artifact=_artifact(content="eval('1')\ndef f():\n    return 1", tests="def test_f():\n    assert f() == 1")
    ))
    assert not result.success
    assert result.output.first_failure.layer.name == "SECURITY"


def test_escalates_after_max_attempts():
    agent = AutomatedValidationAgent()
    result = agent.process(AutomatedValidationRequest(artifact=_artifact(content="eval(x)"), attempt=3, max_attempts=3))
    assert result.escalate
    assert result.escalation_reason == "retry_limit_exceeded"


def test_does_not_escalate_before_max_attempts():
    agent = AutomatedValidationAgent()
    result = agent.process(AutomatedValidationRequest(artifact=_artifact(content="eval(x)"), attempt=1, max_attempts=3))
    assert not result.escalate

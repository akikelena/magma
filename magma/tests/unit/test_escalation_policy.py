from magma.core.governance.escalation import EscalationPolicy, EscalationReason


def test_low_confidence_escalates():
    policy = EscalationPolicy(confidence_threshold=0.7)
    assert policy.evaluate(confidence=0.5) == EscalationReason.LOW_CONFIDENCE


def test_retry_limit_escalates():
    policy = EscalationPolicy(max_retries=3)
    assert policy.evaluate(attempt=4) == EscalationReason.RETRY_LIMIT_EXCEEDED


def test_conflict_escalates():
    policy = EscalationPolicy()
    assert policy.evaluate(has_conflict=True) == EscalationReason.CONFLICTING_SPECIFICATION


def test_security_sensitive_text_escalates():
    policy = EscalationPolicy()
    assert policy.evaluate(affected_text="update the authentication token flow") == EscalationReason.SECURITY_CRITICAL


def test_healthy_case_does_not_escalate():
    policy = EscalationPolicy()
    assert policy.evaluate(confidence=0.95, attempt=1, has_conflict=False, affected_text="rename a field") is None

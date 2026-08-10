"""Escalation policy: the rules that route a case to the Human Expert Agent.

Review is triggered under four conditions: low documentation-interpretation
confidence, validation failures exceeding the retry limit, conflicting
specifications, and security-critical changes (authentication,
authorization, encryption). Encoded here as pure, testable predicates so
every agent applies the same thresholds instead of each hardcoding its own.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class EscalationReason(str, enum.Enum):
    LOW_CONFIDENCE = "low_confidence"
    RETRY_LIMIT_EXCEEDED = "retry_limit_exceeded"
    CONFLICTING_SPECIFICATION = "conflicting_specification"
    SECURITY_CRITICAL = "security_critical"


SECURITY_SENSITIVE_KEYWORDS = (
    "authentication",
    "authorization",
    "encryption",
    "credential",
    "token",
    "password",
    "tls",
    "ssl",
)


@dataclass(frozen=True)
class EscalationPolicy:
    confidence_threshold: float = 0.70
    max_retries: int = 3

    def evaluate(
        self,
        *,
        confidence: float | None = None,
        attempt: int | None = None,
        has_conflict: bool = False,
        affected_text: str = "",
    ) -> EscalationReason | None:
        """Return the first applicable escalation reason, or None."""
        if confidence is not None and confidence < self.confidence_threshold:
            return EscalationReason.LOW_CONFIDENCE
        if attempt is not None and attempt >= self.max_retries:
            return EscalationReason.RETRY_LIMIT_EXCEEDED
        if has_conflict:
            return EscalationReason.CONFLICTING_SPECIFICATION
        lowered = affected_text.lower()
        if any(keyword in lowered for keyword in SECURITY_SENSITIVE_KEYWORDS):
            return EscalationReason.SECURITY_CRITICAL
        return None

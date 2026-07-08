"""Automated Validation Agent.

Modality: deterministic service. Applies six checks in sequence, each
realized through deterministic tooling rather than model judgment, to
ensure reproducible outcomes. Ordered by ascending cost: syntax, style,
functional, integration, security, performance.

Input (pre): a generated template and its test cases.
Output (post): per-layer pass/fail results and, on failure, a rework
directive.

Failure: a failure at any layer returns the template for regeneration with
diagnostics, and the agent permits up to three autonomous rework attempts
before escalating to the Human Expert Agent.
"""
from __future__ import annotations

from dataclasses import dataclass

from magma.core.governance.audit_log import AuditLogger
from magma.core.governance.escalation import EscalationPolicy
from magma.core.interfaces.agent import Agent, AgentResult, Modality
from magma.core.models.execution import GeneratedArtifact
from magma.core.models.validation import LayerResult, ValidationOutcome
from magma.phase3_validation.validators import DEFAULT_VALIDATORS, Validator


@dataclass
class AutomatedValidationRequest:
    artifact: GeneratedArtifact
    attempt: int = 1
    max_attempts: int = 3


class AutomatedValidationAgent(Agent[AutomatedValidationRequest, ValidationOutcome]):
    name = "Automated Validation Agent"
    modality = Modality.ALGORITHMIC

    def __init__(
        self,
        validators: list[Validator] | None = None,
        escalation_policy: EscalationPolicy | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        super().__init__(blackboard=None)
        self.validators = validators or DEFAULT_VALIDATORS
        self.escalation_policy = escalation_policy or EscalationPolicy()
        self.audit = audit_logger or AuditLogger()

    def process(self, data: AutomatedValidationRequest) -> AgentResult[ValidationOutcome]:
        results: list[LayerResult] = []
        for validator in self.validators:
            result = validator.validate(data.artifact)
            results.append(result)
            if not result.passed:
                # Ascending-cost ordering: stop at the first failing layer
                # rather than running the more expensive checks that follow it.
                break

        outcome = ValidationOutcome(layer_results=results, attempt=data.attempt, max_attempts=data.max_attempts)
        escalation_reason = self.escalation_policy.evaluate(attempt=data.attempt if not outcome.passed else None)

        self.audit.log(
            self.name, "validate", episode_id=data.artifact.template_id,
            attempt=data.attempt, passed=outcome.passed,
            failing_layer=outcome.first_failure.layer.name if outcome.first_failure else None,
        )
        return AgentResult(
            agent_name=self.name, modality=self.modality, output=outcome, success=outcome.passed,
            escalate=escalation_reason is not None,
            escalation_reason=escalation_reason.value if escalation_reason else None,
            rationale=outcome.first_failure.details if outcome.first_failure else None,
        )

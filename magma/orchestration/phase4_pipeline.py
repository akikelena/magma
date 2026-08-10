"""Wires the System Improvement Module together: the Explainability,
Usage Feedback, and Adaptive Learning agents. Modeled as
a standalone pipeline class -- consistent with Phase1Pipeline /
Phase2Pipeline / Phase3Pipeline -- rather than being inlined into
`MagmaWorkflow`, so this module too can be run standalone for testing or
partial runs, and so any orchestration layer (procedural `MagmaWorkflow`,
or a CrewAI crew) can dispatch it the same way as the other three
phases.

In production this phase is dispatched from the asynchronous completion
notification emitted by the Code Validation Module, not inline on the
request path; `run()` being a direct call here is what
makes tests deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

from magma.core.governance.audit_log import AuditLogger
from magma.core.interfaces.llm_provider import LLMProvider
from magma.orchestration.phase1_pipeline import Phase1Result
from magma.orchestration.phase2_pipeline import Phase2Result
from magma.orchestration.phase3_pipeline import Phase3Result
from magma.phase4_improvement.adaptive_learning_agent import (
    AdaptiveLearningAgent,
    AdaptiveLearningOutcome,
    AdaptiveLearningRequest,
    BanditReward,
)
from magma.phase4_improvement.explainability_agent import ExplainabilityAgent, ExplanationRecord, ExplanationRequest
from magma.phase4_improvement.prompt_structure import PromptStructureFeatures
from magma.phase4_improvement.usage_feedback_agent import FeedbackReport, UsageFeedbackAgent

# Execution modes used as the Adaptive Learning Agent's bandit arms: the
# routing decision it's learning a policy over.
DEFAULT_ROUTING_ARMS = ["lightweight", "comprehensive", "parallel"]
ROUTING_CONTEXT_DIM = 3  # [change_severity_ordinal / 5, affected_template_count, validation_attempts]


@dataclass
class Phase4Outcome:
    explanation: ExplanationRecord | None = None
    feedback: FeedbackReport | None = None
    adaptive_learning: AdaptiveLearningOutcome | None = None


class Phase4Pipeline:
    """Runs the System Improvement Module end to end."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        audit_logger: AuditLogger | None = None,
        arm_names: list[str] | None = None,
        context_dim: int = ROUTING_CONTEXT_DIM,
    ) -> None:
        self.audit = audit_logger or AuditLogger()
        self.arm_names = arm_names or list(DEFAULT_ROUTING_ARMS)
        self.explainability_agent = ExplainabilityAgent(llm_provider=llm_provider, audit_logger=self.audit)
        self.usage_feedback_agent = UsageFeedbackAgent(audit_logger=self.audit)
        self.adaptive_learning_agent = AdaptiveLearningAgent(
            arm_names=self.arm_names, context_dim=context_dim, audit_logger=self.audit
        )

    def run(
        self,
        changeset_id: str,
        phase1_result: Phase1Result,
        phase2_result: Phase2Result | None,
        phase3_result: Phase3Result | None,
    ) -> Phase4Outcome:
        explanation_result = self.explainability_agent.process(
            ExplanationRequest(
                episode_id=changeset_id,
                audit_records=self.audit.for_episode(changeset_id),
                validation_passed=phase3_result.all_passed if phase3_result else None,
            )
        )
        feedback_result = self.usage_feedback_agent.process(self.audit.all())

        max_severity = phase1_result.change_set.max_severity
        max_severity_ordinal = int(max_severity) if max_severity is not None else 0
        is_security_related = any(
            c.escalation_reason == "security_critical" for c in phase1_result.change_set.changes
        )

        # Supervised half: predict/record the prompt structure used for
        # this changeset's templates, and collect successful episodes for
        # fine-tuning.
        if phase2_result is not None:
            validated_by_template = (
                {v.template_id: v for v in phase3_result.validated} if phase3_result is not None else {}
            )
            structure_features = PromptStructureFeatures(
                change_type_ordinal=max_severity_ordinal,
                affected_entity_count=len(phase1_result.coherence.sequenced_template_ids),
                is_security_related=is_security_related,
            )
            predicted_structure = self.adaptive_learning_agent.predict_prompt_structure(structure_features)
            for artifact in phase2_result.artifacts:
                validated = validated_by_template.get(artifact.template_id)
                success = bool(validated and validated.outcome and validated.outcome.passed)
                self.adaptive_learning_agent.record_prompt_structure_outcome(structure_features, predicted_structure, success)
                self.adaptive_learning_agent.record_episode_outcome(
                    episode_id=changeset_id, template_id=artifact.template_id,
                    prompt=phase2_result.prompts_by_template.get(artifact.template_id, ""),
                    completion=(validated.final_artifact.content if validated and validated.final_artifact else artifact.content),
                    success=success,
                )
            self.adaptive_learning_agent.dispatch_fine_tuning()

        adaptive_outcome = None
        if phase3_result is not None:
            context = [max_severity_ordinal / 5.0, float(len(phase3_result.validated)),
                       float(max((v.attempts for v in phase3_result.validated), default=0))]
            reward = BanditReward(
                success=phase3_result.all_passed,
                time_to_resolution_seconds=60.0 * max((v.attempts for v in phase3_result.validated), default=1),
                human_intervention_count=len(phase3_result.escalated),
            )
            adaptive_result = self.adaptive_learning_agent.process(
                AdaptiveLearningRequest(context=context, observed_arm=self.arm_names[0], observed_reward=reward)
            )
            adaptive_outcome = adaptive_result.output

        return Phase4Outcome(
            explanation=explanation_result.output, feedback=feedback_result.output, adaptive_learning=adaptive_outcome
        )

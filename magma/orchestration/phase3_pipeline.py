"""Wires the Code Validation Module together and realizes the corrective
cycle: on validation failure, the Code Generation Agent reworks the
artifact with the failing layer's diagnostics appended, bounded by a
maximum of three iterations; if it still fails, the case escalates to
the Human Expert Agent rather than looping forever.

The rework-vs-escalate decision itself is delegated to the Task
Orchestration Agent's `evaluate_quality_gate` -- the supervisor owns
that decision, `AutomatedValidationAgent` only reports pass/fail per
layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from magma.core.governance.audit_log import AuditLogger
from magma.core.governance.escalation import EscalationReason
from magma.core.interfaces.llm_provider import LLMProvider
from magma.core.models.execution import GeneratedArtifact
from magma.core.models.validation import ValidationOutcome
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.orchestration.phase2_pipeline import Phase2Result
from magma.phase2_artifact_generation.code_generation_agent import CodeGenerationAgent, CodeGenerationRequest
from magma.phase2_artifact_generation.task_orchestration_agent import TaskOrchestrationAgent
from magma.phase3_validation.automated_validation_agent import AutomatedValidationAgent, AutomatedValidationRequest
from magma.phase3_validation.human_expert_agent import HumanExpertAgent, ReviewTicket


@dataclass
class TemplateValidationResult:
    template_id: str
    final_artifact: GeneratedArtifact | None
    outcome: ValidationOutcome | None
    attempts: int
    escalated: bool
    review_ticket: ReviewTicket | None = None


@dataclass
class Phase3Result:
    changeset_id: str
    validated: list[TemplateValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(v.outcome is not None and v.outcome.passed for v in self.validated)

    @property
    def escalated(self) -> list[TemplateValidationResult]:
        return [v for v in self.validated if v.escalated]


class Phase3Pipeline:
    """Runs the Code Validation Module, including the bounded rework loop."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        knowledge_repository: KnowledgeRepositoryAgent | None = None,
        max_attempts: int = 3,
        human_expert_agent: HumanExpertAgent | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.audit = audit_logger or AuditLogger()
        self.max_attempts = max_attempts
        self.code_generation_agent = CodeGenerationAgent(llm_provider=llm_provider, audit_logger=self.audit)
        self.validation_agent = AutomatedValidationAgent(audit_logger=self.audit)
        # `human_expert_agent` is accepted (rather than always constructed
        # here) so a workflow that spans Phase 1-3 can share one review
        # queue -- see `magma.orchestration.workflow.MagmaWorkflow`, which
        # is where escalations from any module should land, not just Phase 3.
        self.human_expert_agent = human_expert_agent or HumanExpertAgent(audit_logger=self.audit)
        # The rework-vs-escalate call is the Task Orchestration Agent's
        # quality-gate decision, not the validator's; a graph store is
        # required to construct it but is not otherwise used in Phase 3.
        graph_store = knowledge_repository.graph if knowledge_repository is not None else None
        self.task_orchestration_agent = TaskOrchestrationAgent(graph_store=graph_store, audit_logger=self.audit)

    def _validate_with_rework(self, artifact: GeneratedArtifact, prompt_text: str) -> TemplateValidationResult:
        attempt = 1
        current_artifact = artifact
        while True:
            validation_result = self.validation_agent.process(
                AutomatedValidationRequest(artifact=current_artifact, attempt=attempt, max_attempts=self.max_attempts)
            )
            outcome = validation_result.output

            if outcome.passed:
                return TemplateValidationResult(
                    template_id=current_artifact.template_id, final_artifact=current_artifact,
                    outcome=outcome, attempts=attempt, escalated=False,
                )

            should_rework = self.task_orchestration_agent.evaluate_quality_gate(
                attempt=attempt, passed=False, max_iterations=self.max_attempts
            )
            if not should_rework:
                ticket = self.human_expert_agent.enqueue(
                    episode_id=current_artifact.template_id,
                    reason=EscalationReason.RETRY_LIMIT_EXCEEDED,
                    context={"diagnostics": outcome.first_failure.details if outcome.first_failure else "", "attempts": attempt},
                )
                return TemplateValidationResult(
                    template_id=current_artifact.template_id, final_artifact=current_artifact,
                    outcome=outcome, attempts=attempt, escalated=True, review_ticket=ticket,
                )

            # Corrective cycle: rework with the failing layer's diagnostics appended.
            attempt += 1
            rework_result = self.code_generation_agent.process(
                CodeGenerationRequest(
                    template_id=current_artifact.template_id,
                    prompt_text=prompt_text,
                    generation_mode=current_artifact.generation_mode,
                    rework_diagnostics=outcome.first_failure.details if outcome.first_failure else None,
                )
            )
            if not rework_result.success:
                # Regeneration itself failed (e.g. unparseable synthesis
                # output) -- escalate rather than looping on a broken artifact.
                ticket = self.human_expert_agent.enqueue(
                    episode_id=current_artifact.template_id,
                    reason=EscalationReason.RETRY_LIMIT_EXCEEDED,
                    context={"diagnostics": "rework generation failed", "attempts": attempt},
                )
                return TemplateValidationResult(
                    template_id=current_artifact.template_id, final_artifact=current_artifact,
                    outcome=outcome, attempts=attempt, escalated=True, review_ticket=ticket,
                )
            current_artifact = rework_result.output

    def run(self, phase2_result: Phase2Result) -> Phase3Result:
        validated = [
            self._validate_with_rework(artifact, phase2_result.prompts_by_template.get(artifact.template_id, ""))
            for artifact in phase2_result.artifacts
        ]
        self.audit.log(
            "Phase3Pipeline", "run_complete", episode_id=phase2_result.execution_plan.changeset_id,
            validated_count=len(validated), escalated_count=sum(1 for v in validated if v.escalated),
        )
        return Phase3Result(changeset_id=phase2_result.execution_plan.changeset_id, validated=validated)

"""Top-level MAGMA workflow: sequences the three synchronous modules
(Documentation Change Detection, Software Artifact Generation, Code
Validation) and triggers the System Improvement Module asynchronously on
completion.

All modules share one `KnowledgeRepositoryAgent` blackboard instance and
never call each other directly -- this class plays the Task Orchestration
Agent's role of reading blackboard state and dispatching the next stage,
scoped here to the top level (the module-internal Task Orchestration Agent
in `phase2_artifact_generation` plans template-level execution *within*
Phase 2). On successful validation the process terminates; on a change set
flagged during Phase 1 or exhausting Phase 3's corrective-cycle budget, the
workflow stops and returns a status indicating human review is pending,
rather than proceeding.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass

from magma.core.governance.audit_log import AuditLogger
from magma.core.interfaces.llm_provider import LLMProvider
from magma.core.utils.metrics import get_metrics
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.orchestration.phase1_pipeline import Phase1Pipeline, Phase1Result
from magma.orchestration.phase2_pipeline import Phase2Pipeline, Phase2Result
from magma.orchestration.phase3_pipeline import Phase3Pipeline, Phase3Result
from magma.orchestration.phase4_pipeline import DEFAULT_ROUTING_ARMS, Phase4Outcome, Phase4Pipeline, ROUTING_CONTEXT_DIM
from magma.phase1_documentation_detection.documentation_processing_agent import RawDocument
from magma.phase3_validation.human_expert_agent import HumanExpertAgent


class WorkflowStatus(str, enum.Enum):
    COMPLETED = "completed"
    PENDING_HUMAN_REVIEW = "pending_human_review"
    ESCALATED_AFTER_VALIDATION = "escalated_after_validation"


@dataclass
class WorkflowResult:
    status: WorkflowStatus
    changeset_id: str
    phase1: Phase1Result
    phase2: Phase2Result | None = None
    phase3: Phase3Result | None = None
    phase4: Phase4Outcome | None = None


class MagmaWorkflow:
    def __init__(
        self,
        knowledge_repository: KnowledgeRepositoryAgent,
        llm_provider: LLMProvider,
        confidence_threshold: float = 0.70,
        max_validation_attempts: int = 3,
        audit_logger: AuditLogger | None = None,
        metrics: object | None = None,
        human_expert_agent: HumanExpertAgent | None = None,
    ) -> None:
        self.audit = audit_logger or AuditLogger()
        self.knowledge_repository = knowledge_repository
        # Prometheus counters/histograms; defaults to a silently-no-op
        # implementation when prometheus_client isn't installed, via
        # get_metrics(), so this is always safe to record into regardless
        # of environment.
        self.metrics = metrics if metrics is not None else get_metrics()
        # One shared review queue for the whole workflow: every module's
        # escalations -- low-confidence classification and
        # conflicting specs from Phase 1, exhausted rework budget from
        # Phase 3 -- land here, not in module-local queues. Defaults to
        # an in-memory queue; callers running as a long-lived or
        # repeatedly-invoked CLI process should inject one backed by
        # `JsonFileTicketStore` so escalations survive past this process.
        self.human_expert_agent = human_expert_agent or HumanExpertAgent(audit_logger=self.audit)
        self.phase1 = Phase1Pipeline(
            knowledge_repository=knowledge_repository, llm_provider=llm_provider,
            confidence_threshold=confidence_threshold, human_expert_agent=self.human_expert_agent,
            audit_logger=self.audit,
        )
        self.phase2 = Phase2Pipeline(knowledge_repository=knowledge_repository, llm_provider=llm_provider, audit_logger=self.audit)
        self.phase3 = Phase3Pipeline(
            llm_provider=llm_provider, knowledge_repository=knowledge_repository,
            max_attempts=max_validation_attempts, human_expert_agent=self.human_expert_agent,
            audit_logger=self.audit,
        )
        self.phase4 = Phase4Pipeline(
            llm_provider=llm_provider, audit_logger=self.audit,
            arm_names=DEFAULT_ROUTING_ARMS, context_dim=ROUTING_CONTEXT_DIM,
        )

    # Backward-compatible accessors: these agents now live on
    # `self.phase4` (extracted into `Phase4Pipeline` so the System
    # Improvement Module is standalone-runnable like the other three),
    # but are exposed here too since callers previously reached them
    # directly off `MagmaWorkflow`.
    @property
    def explainability_agent(self):
        return self.phase4.explainability_agent

    @property
    def usage_feedback_agent(self):
        return self.phase4.usage_feedback_agent

    @property
    def adaptive_learning_agent(self):
        return self.phase4.adaptive_learning_agent

    def run(
        self, old_document: RawDocument, new_document: RawDocument, template_contents: dict[str, str]
    ) -> WorkflowResult:
        phase1_start = time.perf_counter()
        phase1_result = self.phase1.run(old_document, new_document)
        self.metrics.observe_phase_duration("phase1", time.perf_counter() - phase1_start)
        changeset_id = phase1_result.change_set.changeset_id

        if phase1_result.requires_human_review:
            phase4_outcome = self.phase4.run(changeset_id, phase1_result, phase2_result=None, phase3_result=None)
            self.metrics.record_episode(WorkflowStatus.PENDING_HUMAN_REVIEW.value)
            return WorkflowResult(
                status=WorkflowStatus.PENDING_HUMAN_REVIEW, changeset_id=changeset_id,
                phase1=phase1_result, phase4=phase4_outcome,
            )

        phase2_start = time.perf_counter()
        phase2_result = self.phase2.run(phase1_result, template_contents)
        self.metrics.observe_phase_duration("phase2", time.perf_counter() - phase2_start)

        phase3_start = time.perf_counter()
        phase3_result = self.phase3.run(phase2_result)
        self.metrics.observe_phase_duration("phase3", time.perf_counter() - phase3_start)
        for validated in phase3_result.validated:
            self.metrics.record_validation_outcome(bool(validated.outcome and validated.outcome.passed))

        phase4_start = time.perf_counter()
        phase4_outcome = self.phase4.run(changeset_id, phase1_result, phase2_result, phase3_result)
        self.metrics.observe_phase_duration("phase4", time.perf_counter() - phase4_start)

        status = WorkflowStatus.COMPLETED if phase3_result.all_passed else WorkflowStatus.ESCALATED_AFTER_VALIDATION
        self.metrics.record_episode(status.value)
        self.audit.log(
            "MagmaWorkflow", "run_complete", episode_id=changeset_id, status=status.value,
        )
        return WorkflowResult(
            status=status, changeset_id=changeset_id,
            phase1=phase1_result, phase2=phase2_result, phase3=phase3_result, phase4=phase4_outcome,
        )

    def run_via_crewai(
        self, old_document: RawDocument, new_document: RawDocument, template_contents: dict[str, str]
    ) -> WorkflowResult:
        """Runs the same four phases as `run()`, but scheduled through
        CrewAI (the production coordination layer) instead of this
        method's direct procedural calls -- `magma.crewai_agents.
        build_magma_crew` wraps each phase pipeline's `run` as a CrewAI
        tool, so the underlying agents and their logic are identical
        either way; only the orchestration layer differs. Returns the
        same `WorkflowResult` shape as `run()` so callers don't need to
        care which orchestration mode produced it.

        Requires `pip install crewai`; raises `ImportError` with a clear
        message otherwise (see `build_magma_crew`).
        """
        from magma.crewai_agents.magma_crew import build_magma_crew

        crew = build_magma_crew(self, old_document, new_document, template_contents)
        try:
            crew.kickoff()
        except Exception as exc:
            # A crewai-internal failure (LLM error, malformed tool call,
            # scheduling fault) still needs an audit trail even though it
            # isn't a domain failure this method can route to a review
            # queue -- log it and let the exception propagate, so callers
            # see the same failure `run()` would surface for an
            # equivalent infrastructure fault, but with a record of it.
            self.audit.log("MagmaWorkflow", "run_via_crewai_failed", episode_id="unknown", error=str(exc))
            raise
        state = crew.magma_state

        phase1_result = state["phase1"]
        changeset_id = phase1_result.change_set.changeset_id
        phase2_result = state.get("phase2")
        phase3_result = state.get("phase3")
        phase4_outcome = state.get("phase4")

        if phase1_result.requires_human_review:
            status = WorkflowStatus.PENDING_HUMAN_REVIEW
        elif phase3_result is not None and phase3_result.all_passed:
            status = WorkflowStatus.COMPLETED
        else:
            status = WorkflowStatus.ESCALATED_AFTER_VALIDATION
        self.metrics.record_episode(status.value)
        self.audit.log(
            "MagmaWorkflow", "run_complete_via_crewai", episode_id=changeset_id, status=status.value,
        )

        return WorkflowResult(
            status=status, changeset_id=changeset_id,
            phase1=phase1_result, phase2=phase2_result, phase3=phase3_result, phase4=phase4_outcome,
        )

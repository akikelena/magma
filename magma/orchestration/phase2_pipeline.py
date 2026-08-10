"""Wires the three Software Artifact Generation Module agents together:
Task Orchestration -> Prompt Generation -> Code Generation.

Consumes a `Phase1Result` (this module's sole input contract with Phase 1)
so the two modules stay decoupled: Phase 2 never reaches back into Phase 1
internals, it only reads the coherence outcome that Phase 1 already wrote
to the blackboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from magma.core.governance.audit_log import AuditLogger
from magma.core.interfaces.llm_provider import LLMProvider
from magma.core.models.execution import ExecutionPlan, GeneratedArtifact
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.orchestration.phase1_pipeline import Phase1Result
from magma.phase2_artifact_generation.code_generation_agent import CodeGenerationAgent, CodeGenerationRequest
from magma.phase2_artifact_generation.prompt_generation_agent import PromptGenerationAgent, PromptGenerationRequest
from magma.phase2_artifact_generation.task_orchestration_agent import OrchestrationRequest, TaskOrchestrationAgent


@dataclass
class Phase2Result:
    execution_plan: ExecutionPlan
    artifacts: list[GeneratedArtifact] = field(default_factory=list)
    failed_template_ids: list[str] = field(default_factory=list)
    # template_id -> prompt text used to generate it, so Phase 3 can
    # resubmit the same prompt (plus validation diagnostics) on rework
    # without recomputing exemplar retrieval / context assembly.
    prompts_by_template: dict[str, str] = field(default_factory=dict)


class Phase2Pipeline:
    """Runs the Software Artifact Generation Module end to end."""

    def __init__(
        self,
        knowledge_repository: KnowledgeRepositoryAgent,
        llm_provider: LLMProvider,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.audit = audit_logger or AuditLogger()
        self.knowledge_repository = knowledge_repository
        self.task_orchestration_agent = TaskOrchestrationAgent(
            graph_store=knowledge_repository.graph, audit_logger=self.audit
        )
        self.prompt_generation_agent = PromptGenerationAgent(
            llm_provider=llm_provider, knowledge_repository=knowledge_repository, audit_logger=self.audit
        )
        self.code_generation_agent = CodeGenerationAgent(llm_provider=llm_provider, audit_logger=self.audit)

    def run(self, phase1_result: Phase1Result, template_contents: dict[str, str]) -> Phase2Result:
        """`template_contents` maps template_id -> current source, since the
        pipeline itself does not own a file-system/VCS integration; the
        caller supplies whatever the sequenced templates currently contain.
        """
        if phase1_result.requires_human_review:
            raise RuntimeError(
                "Phase 1 flagged this changeset for human review "
                f"({phase1_result.escalation_reasons}); Phase 2 must not run until it is resolved."
            )

        # The Task Orchestration Agent performs its own affected-artifact
        # and dependency analysis against the knowledge graph; we
        # deliberately do not pass Phase 1's already-sequenced ids here
        # so this stage stays independently correct rather than trusting an
        # upstream agent's internal computation.
        plan_result = self.task_orchestration_agent.process(
            OrchestrationRequest(change_set=phase1_result.change_set)
        )
        plan = plan_result.output

        artifacts: list[GeneratedArtifact] = []
        failed: list[str] = []
        prompts_by_template: dict[str, str] = {}
        for task in plan.ordered_tasks:
            template_content = template_contents.get(task.template_id, "")
            prompt_result = self.prompt_generation_agent.process(
                PromptGenerationRequest(task=task, template_content=template_content)
            )
            prompts_by_template[task.template_id] = prompt_result.output.prompt_text
            generation_result = self.code_generation_agent.process(
                CodeGenerationRequest(
                    template_id=task.template_id,
                    prompt_text=prompt_result.output.prompt_text,
                    generation_mode=task.generation_mode,
                )
            )
            if generation_result.success:
                artifacts.append(generation_result.output)
            else:
                failed.append(task.template_id)

        self.audit.log(
            "Phase2Pipeline", "run_complete", episode_id=plan.changeset_id,
            artifact_count=len(artifacts), failed_count=len(failed),
        )
        return Phase2Result(
            execution_plan=plan, artifacts=artifacts, failed_template_ids=failed,
            prompts_by_template=prompts_by_template,
        )

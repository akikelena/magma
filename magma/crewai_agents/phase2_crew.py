"""CrewAI wiring for the Software Artifact Generation Module.

Same pattern as phase1_crew.py, except the real input is a `Phase1Result`
object rather than plain strings an LLM could plausibly supply, so it's
captured by closure at crew-build time instead of reconstructed from a
tool-call schema -- the caller already has it in hand, same as
`MagmaWorkflow.run` does between phases.
"""
from __future__ import annotations

from typing import Any

from magma.crewai_agents.settings import CrewAgentSettings, CrewSettings, load_agent_settings, load_crew_settings
from magma.orchestration.phase1_pipeline import Phase1Result
from magma.orchestration.phase2_pipeline import Phase2Pipeline, Phase2Result


def build_phase2_crew(
    pipeline: Phase2Pipeline,
    phase1_result: Phase1Result,
    template_contents: dict[str, str],
    agent_settings: CrewAgentSettings | None = None,
    crew_settings: CrewSettings | None = None,
) -> Any:
    """Build a CrewAI `Crew` whose task delegates to `pipeline.run(phase1_result, template_contents)`.

    The real `Phase2Result` is available on `crew.magma_result["result"]`
    after `kickoff()` -- the LLM-facing tool return value is a JSON-safe
    summary, not the full object. `agent_settings`/`crew_settings` default
    to config/crewai.yaml (phase2 / crew sections). Raises `ImportError` if
    crewai isn't installed, so callers can fall back to `pipeline.run(...)`.
    """
    try:
        import crewai
    except ImportError as exc:  # pragma: no cover - exercised only when crewai is absent
        raise ImportError(
            "crewai is not installed; run `pip install crewai` to use the CrewAI-orchestrated "
            "Phase 2 crew, or call magma.orchestration.Phase2Pipeline.run(...) directly instead."
        ) from exc

    agent_settings = agent_settings or load_agent_settings("phase2")
    crew_settings = crew_settings or load_crew_settings()
    holder: dict[str, Phase2Result] = {}

    def _generate_artifacts_tool() -> dict:
        result = pipeline.run(phase1_result, template_contents)
        holder["result"] = result
        return {
            "execution_mode": result.execution_plan.execution_mode.value,
            "task_count": len(result.execution_plan.ordered_tasks),
            "artifacts_generated": [a.template_id for a in result.artifacts],
            "failed_template_ids": result.failed_template_ids,
        }

    software_artifact_generator = crewai.Agent(
        role="Software Artifact Generator",
        goal="Plan the workflow and generate updated template artifacts for the sequenced changeset",
        backstory=(
            "Coordinates the Task Orchestration, Prompt Generation, and Code Generation agents "
            "that make up MAGMA's Software Artifact Generation Module."
        ),
        tools=[_generate_artifacts_tool],
        **agent_settings.as_kwargs(),
    )

    generate_task = crewai.Task(
        description="Run the Phase 2 pipeline over the Phase 1 changeset already detected.",
        agent=software_artifact_generator,
        expected_output="A JSON summary of the execution plan and generated/failed artifacts.",
    )

    crew = crewai.Crew(
        agents=[software_artifact_generator], tasks=[generate_task], process=crewai.Process.sequential,
        **crew_settings.as_kwargs(),
    )
    crew.magma_result = holder
    return crew

"""CrewAI wiring for the Code Validation Module.

Same pattern as phase2_crew.py: the real input (`Phase2Result`) is
captured by closure; `Phase3Pipeline` keeps sole ownership of the
six-layer validation and bounded rework-cycle logic, CrewAI only
schedules the call.
"""
from __future__ import annotations

from typing import Any

from magma.crewai_agents.settings import CrewAgentSettings, CrewSettings, load_agent_settings, load_crew_settings
from magma.orchestration.phase2_pipeline import Phase2Result
from magma.orchestration.phase3_pipeline import Phase3Pipeline, Phase3Result


def build_phase3_crew(
    pipeline: Phase3Pipeline,
    phase2_result: Phase2Result,
    agent_settings: CrewAgentSettings | None = None,
    crew_settings: CrewSettings | None = None,
) -> Any:
    """Build a CrewAI `Crew` whose task delegates to `pipeline.run(phase2_result)`.

    The real `Phase3Result` is available on `crew.magma_result["result"]`
    after `kickoff()`. `agent_settings`/`crew_settings` default to
    config/crewai.yaml (phase3 / crew sections) -- phase3's default
    `max_execution_time` bounds the sandboxed functional-validator
    subprocess chain. Raises `ImportError` if crewai isn't installed, so
    callers can fall back to `pipeline.run(...)` directly.
    """
    try:
        import crewai
    except ImportError as exc:  # pragma: no cover - exercised only when crewai is absent
        raise ImportError(
            "crewai is not installed; run `pip install crewai` to use the CrewAI-orchestrated "
            "Phase 3 crew, or call magma.orchestration.Phase3Pipeline.run(...) directly instead."
        ) from exc

    agent_settings = agent_settings or load_agent_settings("phase3")
    crew_settings = crew_settings or load_crew_settings()
    holder: dict[str, Phase3Result] = {}

    def _validate_artifacts_tool() -> dict:
        result = pipeline.run(phase2_result)
        holder["result"] = result
        return {
            "changeset_id": result.changeset_id,
            "all_passed": result.all_passed,
            "validated_count": len(result.validated),
            "escalated_count": len(result.escalated),
        }

    code_validator = crewai.Agent(
        role="Code Validator",
        goal="Run the six-layer validation pipeline and drive the bounded corrective rework cycle",
        backstory=(
            "Coordinates the Automated Validation Agent's six sequential checks (syntax, style, "
            "functional, integration, security, performance) and the Human Expert Agent escalation "
            "path that make up MAGMA's Code Validation Module."
        ),
        tools=[_validate_artifacts_tool],
        **agent_settings.as_kwargs(),
    )

    validate_task = crewai.Task(
        description="Run the Phase 3 pipeline over the Phase 2 generated artifacts.",
        agent=code_validator,
        expected_output="A JSON summary of the validation outcome and any escalations.",
    )

    crew = crewai.Crew(
        agents=[code_validator], tasks=[validate_task], process=crewai.Process.sequential,
        **crew_settings.as_kwargs(),
    )
    crew.magma_result = holder
    return crew

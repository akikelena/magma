"""CrewAI wiring for the System Improvement Module.

Same pattern as phase2/phase3: the real inputs are captured by closure;
`Phase4Pipeline` keeps sole ownership of the Explainability, Usage
Feedback, and Adaptive Learning agents. Phase 4 runs asynchronously on
episode completion in production, not on the request path, so this is
the crew a completion-notification handler kicks off,
independent of whichever orchestration ran phases 1-3.
"""
from __future__ import annotations

from typing import Any

from magma.crewai_agents.settings import CrewAgentSettings, CrewSettings, load_agent_settings, load_crew_settings
from magma.orchestration.phase1_pipeline import Phase1Result
from magma.orchestration.phase2_pipeline import Phase2Result
from magma.orchestration.phase3_pipeline import Phase3Result
from magma.orchestration.phase4_pipeline import Phase4Outcome, Phase4Pipeline


def build_phase4_crew(
    pipeline: Phase4Pipeline,
    changeset_id: str,
    phase1_result: Phase1Result,
    phase2_result: Phase2Result | None,
    phase3_result: Phase3Result | None,
    agent_settings: CrewAgentSettings | None = None,
    crew_settings: CrewSettings | None = None,
) -> Any:
    """Build a CrewAI `Crew` whose task delegates to `pipeline.run(...)`.

    The real `Phase4Outcome` is available on `crew.magma_result["result"]`
    after `kickoff()`. `agent_settings`/`crew_settings` default to
    config/crewai.yaml (phase4 / crew sections). Raises `ImportError` if
    crewai isn't installed, so callers can fall back to `pipeline.run(...)`.
    """
    try:
        import crewai
    except ImportError as exc:  # pragma: no cover - exercised only when crewai is absent
        raise ImportError(
            "crewai is not installed; run `pip install crewai` to use the CrewAI-orchestrated "
            "Phase 4 crew, or call magma.orchestration.Phase4Pipeline.run(...) directly instead."
        ) from exc

    agent_settings = agent_settings or load_agent_settings("phase4")
    crew_settings = crew_settings or load_crew_settings()
    holder: dict[str, Phase4Outcome] = {}

    def _improve_system_tool() -> dict:
        result = pipeline.run(changeset_id, phase1_result, phase2_result, phase3_result)
        holder["result"] = result
        return {
            "changeset_id": changeset_id,
            "explanation_complete": bool(result.explanation and result.explanation.is_complete),
            "adaptive_learning_recorded": result.adaptive_learning is not None,
        }

    system_improver = crewai.Agent(
        role="System Improver",
        goal="Explain the episode, aggregate usage feedback, and update the adaptive routing policy",
        backstory=(
            "Coordinates the Explainability, Usage Feedback, and Adaptive Learning agents that make "
            "up MAGMA's System Improvement Module, run asynchronously on episode completion."
        ),
        tools=[_improve_system_tool],
        **agent_settings.as_kwargs(),
    )

    improve_task = crewai.Task(
        description="Run the Phase 4 pipeline over the completed episode's phase 1-3 results.",
        agent=system_improver,
        expected_output="A JSON summary of the explanation and adaptive-learning outcome.",
    )

    crew = crewai.Crew(
        agents=[system_improver], tasks=[improve_task], process=crewai.Process.sequential,
        **crew_settings.as_kwargs(),
    )
    crew.magma_result = holder
    return crew

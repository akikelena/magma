"""Top-level CrewAI wiring for the full MAGMA workflow: composes the four
per-phase crews into one `crewai.Crew` under `Process.sequential`, with
task-to-task `context` chaining mirroring `MagmaWorkflow.run`'s control
flow -- including the early exit when Phase 1 flags a changeset for
human review, which skips Phase 2/3 and runs only Phase 4.

Each phase's tool closes over a shared `state` dict rather than taking
LLM-authored arguments, for the same reason phase2_crew.py explains: a
Phase1Result/Phase2Result/Phase3Result isn't something an LLM should be
asked to reconstruct from a tool-call schema. `crew.magma_state` exposes
that dict after `kickoff()` so a caller (`MagmaWorkflow.run_via_crewai`)
gets back the same typed results `MagmaWorkflow.run` would have returned.
"""
from __future__ import annotations

import time
from typing import Any

from magma.crewai_agents.settings import CrewAgentSettings, CrewSettings, load_agent_settings, load_crew_settings
from magma.phase1_documentation_detection.documentation_processing_agent import RawDocument

_PHASES = ("phase1", "phase2", "phase3", "phase4")


def build_magma_crew(
    workflow: "magma.orchestration.workflow.MagmaWorkflow",  # noqa: F821 - string annotation avoids a hard import cycle
    old_document: RawDocument,
    new_document: RawDocument,
    template_contents: dict[str, str],
    agent_settings: dict[str, CrewAgentSettings] | None = None,
    crew_settings: CrewSettings | None = None,
) -> Any:
    """Build a CrewAI `Crew` that runs `workflow`'s four phase pipelines in
    sequence.

    `agent_settings` maps phase name ("phase1".."phase4") to a
    `CrewAgentSettings`; any phase left out defaults to config/crewai.yaml.
    `crew_settings` defaults to config/crewai.yaml's `crew` section. Raises
    `ImportError` if crewai isn't installed, so callers can fall back to
    `workflow.run(...)` directly.
    """
    try:
        import crewai
    except ImportError as exc:  # pragma: no cover - exercised only when crewai is absent
        raise ImportError(
            "crewai is not installed; run `pip install crewai` to use the CrewAI-orchestrated "
            "MAGMA crew, or call magma.orchestration.MagmaWorkflow.run(...) directly instead."
        ) from exc

    agent_settings = agent_settings or {}
    resolved_settings = {phase: agent_settings.get(phase) or load_agent_settings(phase) for phase in _PHASES}
    crew_settings = crew_settings or load_crew_settings()

    state: dict[str, Any] = {}

    def _phase1_tool() -> dict:
        start = time.perf_counter()
        result = workflow.phase1.run(old_document, new_document)
        workflow.metrics.observe_phase_duration("phase1", time.perf_counter() - start)
        state["phase1"] = result
        return {
            "changeset_id": result.change_set.changeset_id,
            "change_count": len(result.change_set.changes),
            "requires_human_review": result.requires_human_review,
        }

    def _phase2_tool() -> dict:
        phase1_result = state["phase1"]
        if phase1_result.requires_human_review:
            return {"skipped": "Phase 1 flagged this changeset for human review; Phase 2 must not run until it is resolved"}
        start = time.perf_counter()
        result = workflow.phase2.run(phase1_result, template_contents)
        workflow.metrics.observe_phase_duration("phase2", time.perf_counter() - start)
        state["phase2"] = result
        return {
            "execution_mode": result.execution_plan.execution_mode.value,
            "artifacts_generated": [a.template_id for a in result.artifacts],
            "failed_template_ids": result.failed_template_ids,
        }

    def _phase3_tool() -> dict:
        if "phase2" not in state:
            return {"skipped": "Phase 2 did not run"}
        start = time.perf_counter()
        result = workflow.phase3.run(state["phase2"])
        workflow.metrics.observe_phase_duration("phase3", time.perf_counter() - start)
        for validated in result.validated:
            workflow.metrics.record_validation_outcome(bool(validated.outcome and validated.outcome.passed))
        state["phase3"] = result
        return {"all_passed": result.all_passed, "validated_count": len(result.validated), "escalated_count": len(result.escalated)}

    def _phase4_tool() -> dict:
        phase1_result = state["phase1"]
        changeset_id = phase1_result.change_set.changeset_id
        start = time.perf_counter()
        result = workflow.phase4.run(changeset_id, phase1_result, state.get("phase2"), state.get("phase3"))
        workflow.metrics.observe_phase_duration("phase4", time.perf_counter() - start)
        state["phase4"] = result
        return {"explanation_complete": bool(result.explanation and result.explanation.is_complete)}

    documentation_change_detector = crewai.Agent(
        role="Documentation Change Detector",
        goal="Detect, classify, and sequence documentation-driven changes for template regeneration",
        backstory="Runs MAGMA's Documentation Change Detection Module (Phase 1).",
        tools=[_phase1_tool], **resolved_settings["phase1"].as_kwargs(),
    )
    software_artifact_generator = crewai.Agent(
        role="Software Artifact Generator",
        goal="Plan the workflow and generate updated template artifacts for the sequenced changeset",
        backstory="Runs MAGMA's Software Artifact Generation Module (Phase 2).",
        tools=[_phase2_tool], **resolved_settings["phase2"].as_kwargs(),
    )
    code_validator = crewai.Agent(
        role="Code Validator",
        goal="Run the six-layer validation pipeline and drive the bounded corrective rework cycle",
        backstory="Runs MAGMA's Code Validation Module (Phase 3).",
        tools=[_phase3_tool], **resolved_settings["phase3"].as_kwargs(),
    )
    system_improver = crewai.Agent(
        role="System Improver",
        goal="Explain the episode, aggregate usage feedback, and update the adaptive routing policy",
        backstory="Runs MAGMA's System Improvement Module (Phase 4).",
        tools=[_phase4_tool], **resolved_settings["phase4"].as_kwargs(),
    )

    phase1_task = crewai.Task(
        description="Run Phase 1 over the supplied old/new documentation versions.",
        agent=documentation_change_detector,
        expected_output="A JSON summary of the detected change set.",
    )
    phase2_task = crewai.Task(
        description="Run Phase 2 over the Phase 1 changeset, unless it was flagged for human review.",
        agent=software_artifact_generator, context=[phase1_task],
        expected_output="A JSON summary of the generated artifacts, or a skip notice.",
    )
    phase3_task = crewai.Task(
        description="Run Phase 3 over the Phase 2 generated artifacts, unless Phase 2 was skipped.",
        agent=code_validator, context=[phase2_task],
        expected_output="A JSON summary of the validation outcome, or a skip notice.",
    )
    phase4_task = crewai.Task(
        description="Run Phase 4 over whatever the episode produced through Phases 1-3.",
        agent=system_improver, context=[phase1_task, phase2_task, phase3_task],
        expected_output="A JSON summary of the explanation and adaptive-learning outcome.",
    )

    crew = crewai.Crew(
        agents=[documentation_change_detector, software_artifact_generator, code_validator, system_improver],
        tasks=[phase1_task, phase2_task, phase3_task, phase4_task],
        process=crewai.Process.sequential,
        **crew_settings.as_kwargs(),
    )
    crew.magma_state = state
    return crew

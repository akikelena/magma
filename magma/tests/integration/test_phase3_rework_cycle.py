"""Integration test: the Phase 3 corrective cycle recovers a failing
artifact within the 3-attempt budget, and escalates to the Human Expert
Agent when it does not."""
from __future__ import annotations

import json

from magma.core.models.execution import ExecutionMode, ExecutionPlan, GeneratedArtifact, GenerationMode
from magma.llm_providers import MockLLMProvider
from magma.orchestration import Phase2Result, Phase3Pipeline


def _plan():
    return ExecutionPlan(changeset_id="cs1", execution_mode=ExecutionMode.LIGHTWEIGHT, ordered_tasks=[])


def _insecure_artifact():
    return GeneratedArtifact(
        template_id="t1", content="eval(x)", test_cases="def test(): pass",
        generation_mode=GenerationMode.TARGETED_MODIFICATION,
    )


def test_recovers_within_budget():
    reasoning_calls = {"n": 0}

    def responder(prompt: str) -> str:
        if "Reasoning from" not in prompt and "template_content" not in prompt:
            reasoning_calls["n"] += 1
            return "reasoning step"
        payload = (
            {"template_content": "eval(x)", "documentation_comments": "", "test_cases": "def test(): pass"}
            if reasoning_calls["n"] < 2
            else {
                "template_content": "def f():\n    return 1",
                "documentation_comments": "fixed",
                "test_cases": "def test():\n    assert f() == 1",
            }
        )
        return json.dumps(payload)

    phase2_result = Phase2Result(execution_plan=_plan(), artifacts=[_insecure_artifact()], prompts_by_template={"t1": "prompt"})
    pipeline = Phase3Pipeline(llm_provider=MockLLMProvider(responder=responder), max_attempts=3)
    result = pipeline.run(phase2_result)

    assert result.all_passed
    assert result.validated[0].attempts <= 3
    assert not result.validated[0].escalated
    assert "eval" not in result.validated[0].final_artifact.content


def test_escalates_after_exhausting_budget():
    def responder(prompt: str) -> str:
        if "Reasoning from" not in prompt and "template_content" not in prompt:
            return "reasoning step"
        return json.dumps({"template_content": "eval(x)", "documentation_comments": "", "test_cases": "def test(): pass"})

    phase2_result = Phase2Result(execution_plan=_plan(), artifacts=[_insecure_artifact()], prompts_by_template={"t1": "prompt"})
    pipeline = Phase3Pipeline(llm_provider=MockLLMProvider(responder=responder), max_attempts=3)
    result = pipeline.run(phase2_result)

    assert not result.all_passed
    assert result.escalated
    ticket = result.escalated[0].review_ticket
    assert ticket is not None
    assert ticket in pipeline.human_expert_agent.pending_tickets()


def test_rework_decision_delegates_to_task_orchestration_quality_gate():
    """Locks in the fix: Phase3Pipeline must consult
    TaskOrchestrationAgent.evaluate_quality_gate for the rework-vs-escalate
    call, not decide it internally."""
    from magma.knowledge_repository import KnowledgeRepositoryAgent

    calls = []

    def responder(prompt: str) -> str:
        if "Reasoning from" not in prompt and "template_content" not in prompt:
            return "reasoning step"
        return json.dumps({"template_content": "eval(x)", "documentation_comments": "", "test_cases": "def test(): pass"})

    kr = KnowledgeRepositoryAgent()
    phase2_result = Phase2Result(execution_plan=_plan(), artifacts=[_insecure_artifact()], prompts_by_template={"t1": "prompt"})
    pipeline = Phase3Pipeline(llm_provider=MockLLMProvider(responder=responder), knowledge_repository=kr, max_attempts=3)

    original = pipeline.task_orchestration_agent.evaluate_quality_gate
    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    pipeline.task_orchestration_agent.evaluate_quality_gate = spy

    pipeline.run(phase2_result)
    assert len(calls) >= 1, "evaluate_quality_gate was never consulted"

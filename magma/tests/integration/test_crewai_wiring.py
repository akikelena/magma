"""Locks in that CrewAI wiring (the production coordination layer) is
actually connected and exercises real MAGMA logic, not a
disconnected demonstration module: each build_phaseN_crew's tool closure
must call the real pipeline and produce the real result object, and
MagmaWorkflow.run_via_crewai must return the same WorkflowResult shape
`run()` would for the same inputs.

Uses the `fake_crewai` fixture (magma/tests/conftest.py) in place of the
real crewai package -- consistent with the rest of this suite never
requiring the heavy optional dependencies (neo4j, weaviate, xgboost,
sklearn, spacy, prometheus_client) to be installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from magma.core.models.change import ChangeSet, ChangeType, DetectedChange
from magma.core.models.documentation import stable_fragment_id
from magma.core.models.execution import ExecutionMode, ExecutionPlan, GeneratedArtifact, GenerationMode
from magma.core.models.graph import EdgeType
from magma.crewai_agents.settings import CrewAgentSettings, CrewSettings, load_agent_settings, load_crew_settings
from magma.knowledge_repository import KnowledgeRepositoryAgent
from magma.llm_providers import MockLLMProvider
from magma.orchestration import MagmaWorkflow, Phase1Pipeline, Phase2Pipeline, Phase2Result, Phase3Pipeline, Phase4Pipeline, WorkflowStatus
from magma.phase1_documentation_detection import RawDocument

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE_URL = "https://milvus.io/docs/search"
HEADING_PATH = ["Milvus Collection.search Reference", "Parameters"]


def _breaking_rename_responder(prompt: str) -> str:
    if "Reasoning from the previous stage" in prompt:
        return json.dumps({
            "template_content": "def search(limit):\n    return limit",
            "documentation_comments": "renamed top_k to limit",
            "test_cases": "def test_search():\n    assert search(5) == 5",
        })
    if "Audit trail:" in prompt:
        return "This episode renamed top_k to limit; validation passed on the first attempt."
    if prompt.strip().startswith("Fragment:"):
        return json.dumps({
            "change_type": "BREAKING", "confidence": 0.95, "intent_shift": 0.9,
            "compatibility_preserved": False, "affected_entities": ["top_k"], "rationale": "renamed",
        })
    return "reasoning: replace top_k with limit"


def _kr_with_template():
    kr = KnowledgeRepositoryAgent()
    kr.add_template_node("search_template", {"path": "search.xtend"})
    kr.link("search_template", stable_fragment_id(SOURCE_URL, HEADING_PATH, 0), EdgeType.GENERATION)
    return kr


def _docs():
    old_doc = RawDocument(content=(FIXTURES / "milvus_search_v2.3.md").read_text(), source_url=SOURCE_URL, vendor="Milvus", version="v2.3.x")
    new_doc = RawDocument(content=(FIXTURES / "milvus_search_v2.4.md").read_text(), source_url=SOURCE_URL, vendor="Milvus", version="v2.4.x")
    return old_doc, new_doc


# -- Without crewai installed: the documented ImportError fallback --------

def test_build_magma_crew_raises_clear_error_without_crewai():
    from magma.crewai_agents.magma_crew import build_magma_crew

    kr = _kr_with_template()
    workflow = MagmaWorkflow(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    old_doc, new_doc = _docs()

    with pytest.raises(ImportError, match="crewai is not installed"):
        build_magma_crew(workflow, old_doc, new_doc, {"search_template": "def search(top_k): ..."})


# -- Per-phase crew builders, each exercising the real pipeline ----------

def test_phase1_crew_tool_calls_real_pipeline(fake_crewai):
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    kr = _kr_with_template()
    pipeline = Phase1Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    old_doc, new_doc = _docs()

    crew = build_phase1_crew(pipeline)
    # classify_document_versions is the direct-input tool: an LLM (or a
    # caller that already has both document texts, no vendor scan
    # needed) supplies content/vendor/version directly. fake_crewai's
    # kickoff() doesn't simulate real LLM-driven tool selection, so call
    # the tool by name to prove it invokes Phase1Pipeline for real.
    tool = crew.agents[0].tools_by_name["_classify_document_versions_tool"]
    result = tool(
        old_content=old_doc.content, new_content=new_doc.content, vendor="Milvus",
        old_version="v2.3.x", new_version="v2.4.x", source_url=SOURCE_URL,
    )
    assert result["change_count"] == 1
    assert result["requires_human_review"] is False
    # The affected template must actually be found: this fails if the
    # tool's constructed RawDocuments don't share source_url with what's
    # already linked in the graph (a real bug this test caught).
    assert result["sequenced_template_ids"] == ["search_template"]


def test_phase1_crew_tool_without_source_url_cannot_match_pre_linked_templates(fake_crewai):
    """Documents the tradeoff: omitting source_url still runs Phase 1
    successfully, but a synthetic fallback URL can never match a
    template already linked (via GENERATION edge) against the
    document's real URL -- so no templates are found and (for content
    with no other conflicts) nothing to reconcile downstream."""
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    kr = _kr_with_template()
    pipeline = Phase1Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    old_doc, new_doc = _docs()

    crew = build_phase1_crew(pipeline)
    tool = crew.agents[0].tools_by_name["_classify_document_versions_tool"]
    result = tool(old_content=old_doc.content, new_content=new_doc.content, vendor="Milvus", old_version="v2.3.x", new_version="v2.4.x")
    assert result["change_count"] == 1
    assert result["sequenced_template_ids"] == []


def test_phase2_crew_tool_calls_real_pipeline_and_exposes_result(fake_crewai):
    from magma.crewai_agents.phase2_crew import build_phase2_crew

    kr = _kr_with_template()
    phase1_pipeline = Phase1Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    old_doc, new_doc = _docs()
    phase1_result = phase1_pipeline.run(old_doc, new_doc)

    phase2_pipeline = Phase2Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    crew = build_phase2_crew(phase2_pipeline, phase1_result, {"search_template": "def search(top_k): ..."})
    outputs = crew.kickoff()

    assert outputs[0]["artifacts_generated"] == ["search_template"]
    assert "result" in crew.magma_result
    assert isinstance(crew.magma_result["result"], Phase2Result)
    assert len(crew.magma_result["result"].artifacts) == 1


def test_phase3_crew_tool_calls_real_pipeline_and_exposes_result(fake_crewai):
    from magma.crewai_agents.phase3_crew import build_phase3_crew

    change_set = ChangeSet(vendor="V", old_version="1", new_version="2", changes=[
        DetectedChange(fragment_id="f1", change_type=ChangeType.MINOR, rationale="x",
                        intent_shift=0.2, compatibility_preserved=True, confidence=0.9)
    ])
    plan = ExecutionPlan(changeset_id=change_set.changeset_id, execution_mode=ExecutionMode.LIGHTWEIGHT, ordered_tasks=[])
    artifact = GeneratedArtifact(
        template_id="t1", content="def f():\n    return 1", test_cases="def test_f():\n    assert f() == 1",
        generation_mode=GenerationMode.TARGETED_MODIFICATION,
    )
    phase2_result = Phase2Result(execution_plan=plan, artifacts=[artifact], prompts_by_template={"t1": "prompt"})

    pipeline = Phase3Pipeline(llm_provider=MockLLMProvider(responder=lambda p: "n/a"))
    crew = build_phase3_crew(pipeline, phase2_result)
    outputs = crew.kickoff()

    assert outputs[0]["all_passed"] is True
    assert crew.magma_result["result"].changeset_id == change_set.changeset_id


def test_phase4_crew_tool_calls_real_pipeline_and_exposes_result(fake_crewai):
    from magma.crewai_agents.phase4_crew import build_phase4_crew

    kr = _kr_with_template()
    phase1_pipeline = Phase1Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    old_doc, new_doc = _docs()
    phase1_result = phase1_pipeline.run(old_doc, new_doc)

    pipeline = Phase4Pipeline(llm_provider=MockLLMProvider(responder=lambda p: "an explanation"))
    crew = build_phase4_crew(pipeline, phase1_result.change_set.changeset_id, phase1_result, None, None)
    outputs = crew.kickoff()

    assert outputs[0]["changeset_id"] == phase1_result.change_set.changeset_id
    assert crew.magma_result["result"].explanation is not None


# -- Top-level composed crew + MagmaWorkflow.run_via_crewai ---------------

def test_run_via_crewai_matches_run_for_a_clean_completion(fake_crewai):
    kr = _kr_with_template()
    workflow = MagmaWorkflow(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    old_doc, new_doc = _docs()

    result = workflow.run_via_crewai(old_doc, new_doc, template_contents={"search_template": "def search(top_k): ..."})

    assert result.status == WorkflowStatus.COMPLETED
    assert result.phase1.change_set.max_severity.name == "BREAKING"
    assert result.phase2 is not None and len(result.phase2.artifacts) == 1
    assert result.phase3 is not None and result.phase3.all_passed
    assert result.phase4 is not None and result.phase4.explanation is not None


def test_run_via_crewai_skips_phase2_and_3_on_human_review(fake_crewai):
    kr = KnowledgeRepositoryAgent()

    def low_confidence(prompt: str) -> str:
        if prompt.strip().startswith("Fragment:"):
            return json.dumps({
                "change_type": "MAJOR", "confidence": 0.2, "intent_shift": 0.3,
                "compatibility_preserved": True, "affected_entities": [], "rationale": "ambiguous",
            })
        return "explanation" if "Audit trail:" in prompt else "n/a"

    workflow = MagmaWorkflow(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=low_confidence))
    old_doc = RawDocument(content="# T\n\nold text\n", source_url="u", vendor="V", version="old")
    new_doc = RawDocument(content="# T\n\nnew text\n", source_url="u", vendor="V", version="new")

    result = workflow.run_via_crewai(old_doc, new_doc, template_contents={})

    assert result.status == WorkflowStatus.PENDING_HUMAN_REVIEW
    assert result.phase2 is None
    assert result.phase3 is None
    assert result.phase4 is not None


# -- Configurable CrewAI Agent/Crew attributes (config/crewai.yaml) -------

def test_default_agent_settings_come_from_the_committed_crewai_yaml():
    phase3_settings = load_agent_settings("phase3")
    assert phase3_settings.max_execution_time == 120  # bounds the sandboxed functional-validator subprocess chain
    assert phase3_settings.max_iter == 20
    assert phase3_settings.max_retry_limit == 2

    phase2_settings = load_agent_settings("phase2")
    assert phase2_settings.max_iter == 25

    crew_defaults = load_crew_settings()
    assert crew_defaults.cache is True
    assert crew_defaults.memory is False


def test_phase1_crew_agent_receives_settings_from_config(fake_crewai):
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    pipeline = Phase1Pipeline(knowledge_repository=_kr_with_template(), llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    crew = build_phase1_crew(pipeline)

    agent = crew.agents[0]
    assert agent.settings["max_iter"] == 20
    assert agent.settings["cache"] is True
    assert agent.settings["allow_delegation"] is False
    assert crew.settings["cache"] is True


def test_phase1_crew_accepts_explicit_settings_overriding_config(fake_crewai):
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    pipeline = Phase1Pipeline(knowledge_repository=_kr_with_template(), llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    custom_agent_settings = CrewAgentSettings(max_iter=99, allow_delegation=True, verbose=True)
    custom_crew_settings = CrewSettings(memory=True, verbose=True)

    crew = build_phase1_crew(pipeline, agent_settings=custom_agent_settings, crew_settings=custom_crew_settings)

    assert crew.agents[0].settings["max_iter"] == 99
    assert crew.agents[0].settings["allow_delegation"] is True
    assert crew.settings["memory"] is True


def test_magma_crew_applies_per_phase_settings_with_config_fallback(fake_crewai):
    from magma.crewai_agents.magma_crew import build_magma_crew

    kr = _kr_with_template()
    workflow = MagmaWorkflow(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    old_doc, new_doc = _docs()

    crew = build_magma_crew(
        workflow, old_doc, new_doc, {"search_template": "def search(top_k): ..."},
        agent_settings={"phase3": CrewAgentSettings(max_execution_time=999)},
    )

    phase1_agent, phase2_agent, phase3_agent, phase4_agent = crew.agents
    # phase1/2/4 fell back to config/crewai.yaml; phase3 used the explicit override.
    assert phase1_agent.settings["max_iter"] == 20
    assert phase2_agent.settings["max_iter"] == 25
    assert phase3_agent.settings["max_execution_time"] == 999
    assert phase4_agent.settings["max_iter"] == 15


def test_crew_agent_settings_from_dict_ignores_unknown_keys():
    settings = CrewAgentSettings.from_dict({"max_iter": 5, "some_future_crewai_field": "ignored"})
    assert settings.max_iter == 5


def test_crew_agent_settings_as_kwargs_omits_unset_optional_fields():
    settings = CrewAgentSettings()
    kwargs = settings.as_kwargs()
    assert "max_rpm" not in kwargs  # None by default -- must not be passed to crewai.Agent as an explicit None
    assert "max_execution_time" not in kwargs
    assert kwargs["max_iter"] == 20


def test_run_via_crewai_logs_and_reraises_on_kickoff_failure(fake_crewai, monkeypatch):
    """A crewai-internal failure (LLM error, scheduling fault) isn't a
    domain failure this method can route to human review, but it must
    still leave an audit trail rather than vanishing silently."""
    kr = _kr_with_template()
    workflow = MagmaWorkflow(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    old_doc, new_doc = _docs()

    def _boom(self, planner=None):
        raise RuntimeError("simulated crewai scheduling failure")

    monkeypatch.setattr(fake_crewai.Crew, "kickoff", _boom)

    with pytest.raises(RuntimeError, match="simulated crewai scheduling failure"):
        workflow.run_via_crewai(old_doc, new_doc, template_contents={"search_template": "def search(top_k): ..."})

    failure_records = [r for r in workflow.audit.all() if r.action == "run_via_crewai_failed"]
    assert len(failure_records) == 1
    assert "simulated crewai scheduling failure" in failure_records[0].details["error"]

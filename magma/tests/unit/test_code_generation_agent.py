import json

from magma.core.models.execution import GenerationMode
from magma.llm_providers import MockLLMProvider
from magma.phase2_artifact_generation.code_generation_agent import (
    CodeGenerationAgent,
    CodeGenerationRequest,
)


def _synth_responder(payload):
    calls = {"n": 0}

    def responder(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return "step-by-step reasoning about the change"
        return "final answer:\n" + json.dumps(payload)

    return responder


def test_successful_generation_returns_artifact():
    payload = {"template_content": "new code", "documentation_comments": "doc", "test_cases": "tests"}
    llm = MockLLMProvider(responder=_synth_responder(payload))
    agent = CodeGenerationAgent(llm_provider=llm)
    result = agent.process(
        CodeGenerationRequest(template_id="t1", prompt_text="do it", generation_mode=GenerationMode.TARGETED_MODIFICATION)
    )
    assert result.success
    assert result.output.content == "new code"
    assert len(llm.calls) == 2  # reasoning stage + synthesis stage (prompt chaining)


def test_unparseable_synthesis_output_fails_cleanly():
    llm = MockLLMProvider(responder=lambda p: "no json here")
    agent = CodeGenerationAgent(llm_provider=llm)
    result = agent.process(
        CodeGenerationRequest(template_id="t1", prompt_text="do it", generation_mode=GenerationMode.TARGETED_MODIFICATION)
    )
    assert not result.success
    assert result.output is None


def test_rework_diagnostics_included_in_reasoning_prompt():
    payload = {"template_content": "fixed", "documentation_comments": "", "test_cases": ""}
    llm = MockLLMProvider(responder=_synth_responder(payload))
    agent = CodeGenerationAgent(llm_provider=llm)
    agent.process(CodeGenerationRequest(
        template_id="t1", prompt_text="do it", generation_mode=GenerationMode.COMPREHENSIVE_SYNTHESIS,
        rework_diagnostics="syntax error on line 3",
    ))
    assert "syntax error on line 3" in llm.calls[0]

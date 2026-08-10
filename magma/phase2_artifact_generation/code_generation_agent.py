"""Code Generation Agent.

Modality: autonomous LLM agent. Synthesizes the modified template using
three prompting techniques: chain-of-thought (step-by-step reasoning),
prompt chaining (decomposing the modification into sequential subtasks
whose outputs feed forward), and expert prompting (specialized reasoning
for domain-specific challenges like template-inheritance resolution or
database-specific query syntax).

Operates in one of two modes selected by the Task Orchestration Agent
(via `TemplateTask.generation_mode`): targeted modification (edit specific
sections, <30% of lines affected) or comprehensive synthesis (regenerate
the entire template).

Failure: generated artifacts that fail validation are returned with
diagnostics for autonomous rework (handled by the caller, which resubmits
with the diagnostics appended to the prompt).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import time

from magma.core.governance.audit_log import AuditLogger
from magma.core.interfaces.agent import Agent, AgentResult, Modality
from magma.core.interfaces.llm_provider import LLMProvider
from magma.core.models.execution import GeneratedArtifact, GenerationMode

_EXPERT_HINTS = {
    "xtend": "You are an expert in Xtend template inheritance and code-generation idioms.",
    "sql": "You are an expert in database-specific query syntax and dialect differences.",
}

_REASONING_SYSTEM_PROMPT = (
    "You are the reasoning stage of a template code-generation pipeline. Think step by step "
    "about how to apply the requested change to the template, including edge cases and how the "
    "template's existing structure constrains the edit. Do not emit code yet."
)

_SYNTHESIS_SYSTEM_PROMPT_TEMPLATE = """{expert_hint}
Using the reasoning below, emit the final artifact as a JSON object on its own line:
{{"template_content": "...", "documentation_comments": "...", "test_cases": "..."}}
Escape newlines within JSON string values. Do not include anything after the JSON object.
"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class CodeGenerationRequest:
    template_id: str
    prompt_text: str
    generation_mode: GenerationMode
    target_language: str = "xtend"
    rework_diagnostics: str | None = None


class CodeGenerationParseError(ValueError):
    pass


class CodeGenerationAgent(Agent[CodeGenerationRequest, GeneratedArtifact]):
    name = "Code Generation Agent"
    modality = Modality.LLM

    def __init__(self, llm_provider: LLMProvider, audit_logger: AuditLogger | None = None) -> None:
        super().__init__(blackboard=None)
        self.llm = llm_provider
        self.audit = audit_logger or AuditLogger()

    def _reasoning_stage(self, request: CodeGenerationRequest) -> str:
        """Chain-of-thought stage: reason before emitting code."""
        prompt = request.prompt_text
        if request.rework_diagnostics:
            prompt += f"\n\n## Prior attempt failed validation with:\n{request.rework_diagnostics}"
        response = self.llm.complete(prompt, system=_REASONING_SYSTEM_PROMPT, temperature=0.0)
        return response.text

    def _synthesis_stage(self, request: CodeGenerationRequest, reasoning: str) -> GeneratedArtifact:
        """Prompt chaining: the reasoning stage's output feeds this stage.
        Expert prompting: a domain-specific system-prompt hint is added
        based on the target language/template-inheritance concerns."""
        expert_hint = _EXPERT_HINTS.get(request.target_language, "You are an expert software engineer.")
        system_prompt = _SYNTHESIS_SYSTEM_PROMPT_TEMPLATE.format(expert_hint=expert_hint)
        synthesis_prompt = (
            f"Requested change and context:\n{request.prompt_text}\n\n"
            f"Reasoning from the previous stage:\n{reasoning}\n\n"
            f"Generation mode: {request.generation_mode.value}."
        )
        response = self.llm.complete(synthesis_prompt, system=system_prompt, temperature=0.0)

        match = _JSON_OBJECT_RE.search(response.text)
        if not match:
            raise CodeGenerationParseError(f"no JSON artifact found in synthesis output: {response.text!r}")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise CodeGenerationParseError(f"malformed JSON artifact: {exc}") from exc

        return GeneratedArtifact(
            template_id=request.template_id,
            content=str(payload.get("template_content", "")),
            documentation_comments=str(payload.get("documentation_comments", "")),
            test_cases=str(payload.get("test_cases", "")),
            generation_mode=request.generation_mode,
        )

    def process(self, data: CodeGenerationRequest) -> AgentResult[GeneratedArtifact]:
        start = time.perf_counter()
        reasoning = self._reasoning_stage(data)
        try:
            artifact = self._synthesis_stage(data, reasoning)
        except CodeGenerationParseError as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.audit.log(
                self.name, "synthesis_parse_failure", episode_id=data.template_id,
                error=str(exc), duration_ms=duration_ms,
            )
            return AgentResult(
                agent_name=self.name, modality=self.modality, output=None, success=False, rationale=str(exc)
            )
        duration_ms = (time.perf_counter() - start) * 1000.0

        self.audit.log(
            self.name, "generate_template", episode_id=data.template_id,
            generation_mode=data.generation_mode.value, is_rework=data.rework_diagnostics is not None,
            duration_ms=duration_ms,
        )
        return AgentResult(agent_name=self.name, modality=self.modality, output=artifact, success=True)

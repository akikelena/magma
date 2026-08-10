"""Prompt Generation Agent.

Modality: autonomous LLM agent. Assembles a context-rich synthesis prompt
for the Code Generation Agent from several sources: annotated template
context (Knowledge Repository), a compressed statement of intent (LLM
call), integrated metadata (languages, conventions, standards, security
policy), and exemplars retrieved by semantic similarity.

Failure: if no sufficiently similar exemplar is retrieved, the agent
proceeds with the constraint specification alone and records the absence
for the Usage Feedback Agent.
"""
from __future__ import annotations

from dataclasses import dataclass

from magma.core.governance.audit_log import AuditLogger
from magma.core.utils.timing import timed
from magma.core.interfaces.agent import Agent, AgentResult, Modality
from magma.core.interfaces.llm_provider import LLMProvider
from magma.core.models.execution import TemplateTask
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent

_COMPRESSION_SYSTEM_PROMPT = (
    "You compress a verbose maintenance instruction into one concise sentence stating the "
    "concrete code change required (e.g. 'add an optional retry_count parameter with default "
    "value 3'). Respond with only that sentence, no preamble."
)


@dataclass
class PromptGenerationRequest:
    task: TemplateTask
    template_content: str
    target_language: str = "xtend"
    framework_conventions: str = ""
    coding_standards: str = ""
    security_policy: str = ""
    exemplar_query: str | None = None
    max_exemplars: int = 3
    token_budget: int = 6000


@dataclass
class GeneratedPrompt:
    template_id: str
    prompt_text: str
    used_exemplar_count: int
    exemplar_absent: bool = False


class PromptGenerationAgent(Agent[PromptGenerationRequest, GeneratedPrompt]):
    name = "Prompt Generation Agent"
    modality = Modality.LLM

    def __init__(
        self,
        llm_provider: LLMProvider,
        knowledge_repository: KnowledgeRepositoryAgent | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        super().__init__(blackboard=knowledge_repository)
        self.llm = llm_provider
        self.knowledge_repository = knowledge_repository
        self.audit = audit_logger or AuditLogger()

    def _compress_intent(self, task: TemplateTask) -> str:
        response = self.llm.complete(task.instructions, system=_COMPRESSION_SYSTEM_PROMPT, temperature=0.0)
        return response.text.strip() or task.instructions

    def _annotated_template_context(self, task: TemplateTask) -> str | None:
        """Pull the template's own graph-node annotations (description,
        language, prior usage notes) from the Knowledge Repository, so the
        prompt carries that context as a distinct input rather than
        relying on the template body alone to convey intent."""
        if self.knowledge_repository is None:
            return None
        node = self.knowledge_repository.graph.get_node(task.template_id)
        if node is None:
            return None
        notes = {
            k: v for k, v in node.properties.items()
            if k not in {"generation_rules", "output_formats", "architectural_constraints"} and v
        }
        if not notes:
            return None
        return "\n".join(f"- {k}: {v}" for k, v in sorted(notes.items()))

    def _resolve_exemplar_query(self, request: PromptGenerationRequest) -> str | None:
        # Explicit caller query wins; otherwise fall back to the compressed
        # task intent so exemplar retrieval isn't purely opt-in and every
        # request benefits from semantic-similarity search when the
        # Knowledge Repository is available.
        if request.exemplar_query is not None:
            return request.exemplar_query
        return request.task.instructions or None

    def _retrieve_exemplars(self, request: PromptGenerationRequest) -> list[str]:
        if self.knowledge_repository is None:
            return []
        query = self._resolve_exemplar_query(request)
        if query is None:
            return []
        try:
            matches = self.knowledge_repository.find_similar(query, top_k=request.max_exemplars)
        except RuntimeError:
            return []  # no embedding client configured; proceed without exemplars
        return [str(m.metadata.get("content", "")) for m in matches if m.metadata.get("content")]

    @staticmethod
    def _fit_to_budget(sections: list[str], token_budget: int) -> str:
        """Greedily includes sections (most important first) while staying
        under a rough whitespace-token budget."""
        included: list[str] = []
        used = 0
        for section in sections:
            tokens = len(section.split())
            if used + tokens > token_budget and included:
                continue
            included.append(section)
            used += tokens
        return "\n\n".join(included)

    def process(self, data: PromptGenerationRequest) -> AgentResult[GeneratedPrompt]:
        with timed() as elapsed:
            intent = self._compress_intent(data.task)
        exemplars = self._retrieve_exemplars(data)
        exemplar_absent = not exemplars

        annotated_context = self._annotated_template_context(data.task)

        sections = [
            f"## Task\n{intent}",
            f"## Target template ({data.task.template_id}, {data.target_language})\n```\n{data.template_content}\n```",
        ]
        if annotated_context:
            sections.append(f"## Annotated template context (Knowledge Repository)\n{annotated_context}")
        if data.task.generation_rules:
            sections.append("## Generation rules\n" + "\n".join(f"- {r}" for r in data.task.generation_rules))
        if data.task.output_formats:
            sections.append("## Required output formats\n" + ", ".join(data.task.output_formats))
        if data.task.architectural_constraints:
            sections.append(f"## Architectural constraints\n{data.task.architectural_constraints}")
        if data.framework_conventions:
            sections.append(f"## Framework conventions\n{data.framework_conventions}")
        if data.coding_standards:
            sections.append(f"## Coding standards\n{data.coding_standards}")
        if data.security_policy:
            sections.append(f"## Security policy\n{data.security_policy}")
        for i, exemplar in enumerate(exemplars, start=1):
            sections.append(f"## Exemplar {i}\n{exemplar}")

        prompt_text = self._fit_to_budget(sections, data.token_budget)

        if exemplar_absent:
            self.audit.log(self.name, "no_exemplar_found", episode_id=data.task.template_id)

        self.audit.log(
            self.name, "generate_prompt", episode_id=data.task.template_id,
            exemplar_count=len(exemplars), prompt_chars=len(prompt_text), duration_ms=elapsed.ms,
        )
        return AgentResult(
            agent_name=self.name, modality=self.modality, success=True,
            output=GeneratedPrompt(
                template_id=data.task.template_id, prompt_text=prompt_text,
                used_exemplar_count=len(exemplars), exemplar_absent=exemplar_absent,
            ),
        )

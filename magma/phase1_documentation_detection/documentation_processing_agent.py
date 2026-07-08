"""Documentation Processing Agent.

Modality: deterministic service.
Input (pre): raw documentation retrieved from HTTP endpoints,
             version-controlled repositories, or APIs.
Output (post): structured fragments, each carrying a semantic label.

Three-stage pipeline: (1) structure detection + parsing into a normalized
tree, (2) NER/dependency parsing/topic modeling, (3) fragment labeling.
Malformed or unparseable sources are recorded and skipped rather than
passed downstream, so a parsing failure cannot propagate into
classification -- this agent realizes part of the first line of defense.
"""
from __future__ import annotations

from dataclasses import dataclass

from magma.core.governance.audit_log import AuditLogger
from magma.core.interfaces.agent import Agent, AgentResult, Modality
from magma.core.models.documentation import DocumentationFragment, stable_fragment_id
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.phase1_documentation_detection.fragment_labeler import label_fragment
from magma.phase1_documentation_detection.nlp_pipeline import HeuristicNLPBackend, NLPBackend
from magma.phase1_documentation_detection.source_parser import parse_source


@dataclass
class RawDocument:
    content: str
    source_url: str
    vendor: str
    version: str


@dataclass
class SkippedSource:
    source_url: str
    reason: str


class DocumentationProcessingAgent(Agent[RawDocument, list[DocumentationFragment]]):
    name = "Documentation Processing Agent"
    modality = Modality.ALGORITHMIC

    def __init__(
        self,
        knowledge_repository: KnowledgeRepositoryAgent | None = None,
        nlp_backend: NLPBackend | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        super().__init__(blackboard=knowledge_repository)
        self.knowledge_repository = knowledge_repository
        self.nlp = nlp_backend or HeuristicNLPBackend()
        self.audit = audit_logger or AuditLogger()
        self.skipped: list[SkippedSource] = []

    def process(self, data: RawDocument) -> AgentResult[list[DocumentationFragment]]:
        try:
            _, sections = parse_source(data.content)
        except Exception as exc:  # noqa: BLE001
            skip = SkippedSource(source_url=data.source_url, reason=f"parse failure: {exc}")
            self.skipped.append(skip)
            self.audit.log(self.name, "skip_source", episode_id=data.source_url, reason=skip.reason)
            return AgentResult(
                agent_name=self.name, modality=self.modality, output=None, success=False,
                rationale=skip.reason,
            )

        if not sections:
            skip = SkippedSource(source_url=data.source_url, reason="no parsable content")
            self.skipped.append(skip)
            self.audit.log(self.name, "skip_source", episode_id=data.source_url, reason=skip.reason)
            return AgentResult(
                agent_name=self.name, modality=self.modality, output=None, success=False,
                rationale=skip.reason,
            )

        fragments: list[DocumentationFragment] = []
        ordinal_by_heading: dict[tuple, int] = {}
        for section in sections:
            heading_key = tuple(section.heading_path)
            ordinal = ordinal_by_heading.get(heading_key, 0)
            ordinal_by_heading[heading_key] = ordinal + 1

            annotation = self.nlp.annotate(section.content)
            label = label_fragment(section.content, annotation)
            fragment = DocumentationFragment(
                content=section.content,
                label=label,
                source_url=data.source_url,
                vendor=data.vendor,
                version=data.version,
                heading_path=section.heading_path,
                fragment_id=stable_fragment_id(data.source_url, section.heading_path, ordinal),
                entities=[e.__dict__ for e in annotation.entities],
                relations=[r.__dict__ for r in annotation.relations],
                is_code_block=section.is_code_block,
            )
            fragments.append(fragment)
            if self.knowledge_repository is not None:
                self.knowledge_repository.add_fragment_node(
                    fragment.fragment_id, fragment.to_dict(), attribution=self.name
                )

        self.audit.log(
            self.name, "process_source", episode_id=data.source_url,
            fragment_count=len(fragments), vendor=data.vendor, version=data.version,
        )
        return AgentResult(agent_name=self.name, modality=self.modality, output=fragments, success=True)

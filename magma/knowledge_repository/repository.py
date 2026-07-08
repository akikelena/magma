"""Knowledge Repository Agent: the blackboard itself.

Modality: deterministic data-management service. This agent is read and
written by every other agent in the architecture; it is not a passive
database but the coordination medium of the blackboard pattern. It composes
a `GraphStore` and a `VectorStore` and adds two properties on top:
timestamped/attributed versioning, and rollback to a prior versioned
state on detection of a downstream error.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from magma.core.governance.audit_log import AuditLogger
from magma.core.interfaces.agent import Agent, AgentResult, Modality
from magma.core.interfaces.message_bus import MessageBus
from magma.core.models.graph import EdgeType, GraphEdge, GraphNode, NodeType
from magma.core.models.template import TemplateComponent
from magma.knowledge_repository.embeddings import EmbeddingClient
from magma.knowledge_repository.graph_store import GraphStore, InMemoryGraphStore
from magma.knowledge_repository.vector_store import (
    DEFAULT_SIMILARITY_THRESHOLD,
    InMemoryVectorStore,
    SimilarityMatch,
    VectorStore,
)


@dataclass
class WriteRequest:
    key: str
    value: Any
    attribution: str | None = None


class KnowledgeRepositoryAgent(Agent[WriteRequest, str], MessageBus):
    """The blackboard: versioned key/value store plus graph and vector layers."""

    name = "Knowledge Repository Agent"
    modality = Modality.ALGORITHMIC

    def __init__(
        self,
        graph_store: GraphStore | None = None,
        vector_store: VectorStore | None = None,
        embedding_client: EmbeddingClient | None = None,
        audit_logger: AuditLogger | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        super().__init__(blackboard=None)
        self.graph = graph_store or InMemoryGraphStore()
        self.vectors = vector_store or InMemoryVectorStore()
        self.embeddings = embedding_client
        self.audit = audit_logger or AuditLogger()
        self.similarity_threshold = similarity_threshold
        # key -> ordered list of (version_id, value) for MessageBus.write/read/rollback
        self._store: dict[str, list[tuple[str, Any]]] = {}

    # -- MessageBus (generic blackboard key/value access) --------------------

    def write(self, key: str, value: Any, *, version: bool = True) -> str:
        history = self._store.setdefault(key, [])
        version_id = f"v{len(history) + 1}"
        history.append((version_id, copy.deepcopy(value)))
        self.audit.log(self.name, "write", episode_id=key, version=version_id)
        return version_id

    def read(self, key: str, *, version: str | None = None) -> Any:
        history = self._store.get(key)
        if not history:
            return None
        if version is None:
            return history[-1][1]
        for vid, value in history:
            if vid == version:
                return value
        raise KeyError(f"no such version {version!r} for key {key!r}")

    def history(self, key: str):
        return [vid for vid, _ in self._store.get(key, [])]

    def rollback(self, key: str, version: str) -> None:
        """Restore `key` to `version` by appending it as the newest revision.

        Appending (rather than truncating) preserves the full audit trail
        required for the Explainability Agent, while `read(key)` reflects
        the rolled-back state immediately.
        """
        value = self.read(key, version=version)
        history = self._store.setdefault(key, [])
        new_version_id = f"v{len(history) + 1}"
        history.append((new_version_id, copy.deepcopy(value)))
        self.audit.log(self.name, "rollback", episode_id=key, restored_from=version, new_version=new_version_id)

    # -- Graph layer convenience wrappers -------------------------------------

    def add_fragment_node(self, fragment_id: str, properties: dict[str, Any], attribution: str | None = None) -> str:
        return self.graph.add_node(
            GraphNode(node_type=NodeType.DOCUMENTATION_FRAGMENT, properties=properties, node_id=fragment_id, attribution=attribution)
        )

    def add_template_node(self, template_id: str, properties: dict[str, Any], attribution: str | None = None) -> str:
        return self.graph.add_node(
            GraphNode(node_type=NodeType.TEMPLATE_COMPONENT, properties=properties, node_id=template_id, attribution=attribution)
        )

    def add_template_component(self, component: TemplateComponent, attribution: str | None = None) -> str:
        """Registers a `TemplateComponent` as a TEMPLATE_COMPONENT graph
        node, the structured counterpart to `add_template_node`'s raw
        properties dict. `component.generation_rules`/`output_formats`
        land in node properties under the same keys Task Orchestration
        Agent's affected-artifact analysis already reads
        (`_template_metadata` in task_orchestration_agent.py), so a
        template registered this way is immediately usable by that
        traversal without any further wiring."""
        return self.add_template_node(
            component.template_id,
            {
                "path": component.path,
                "language": component.language,
                "content": component.content,
                "generation_rules": component.generation_rules,
                "output_formats": component.output_formats,
            },
            attribution=attribution,
        )

    def add_concept_node(self, concept_id: str, properties: dict[str, Any], attribution: str | None = None) -> str:
        return self.graph.add_node(
            GraphNode(node_type=NodeType.SEMANTIC_CONCEPT, properties=properties, node_id=concept_id, attribution=attribution)
        )

    def link(self, source_id: str, target_id: str, edge_type: EdgeType, **properties: Any) -> None:
        self.graph.add_edge(GraphEdge(source_id=source_id, target_id=target_id, edge_type=edge_type, properties=properties))

    # -- Vector layer convenience wrappers ------------------------------------

    def index_fragment(self, fragment_id: str, text: str, metadata: dict[str, Any]) -> None:
        if self.embeddings is None:
            raise RuntimeError("no EmbeddingClient configured on this KnowledgeRepositoryAgent")
        vector = self.embeddings.embed([text])[0]
        self.vectors.upsert(fragment_id, vector, metadata)

    def find_similar(self, text: str, *, top_k: int = 10, threshold: float | None = None) -> list[SimilarityMatch]:
        if self.embeddings is None:
            raise RuntimeError("no EmbeddingClient configured on this KnowledgeRepositoryAgent")
        vector = self.embeddings.embed([text])[0]
        return self.vectors.similarity_search(vector, top_k=top_k, threshold=threshold or self.similarity_threshold)

    # -- Agent interface -------------------------------------------------------

    def process(self, data: WriteRequest) -> AgentResult[str]:
        try:
            version_id = self.write(data.key, data.value)
            return AgentResult(agent_name=self.name, modality=self.modality, output=version_id, success=True)
        except Exception as exc:  # noqa: BLE001 - convert any failure into a result, not a raise
            return AgentResult(
                agent_name=self.name, modality=self.modality, output=None, success=False,
                escalate=False, rationale=f"write failed: {exc}",
            )

"""Knowledge graph node/edge models for the Knowledge Repository.

Three node types and four directed edge types: dependency (one
component requires another), generation (a template produces an
artifact), supersession (a newer version replaces an earlier one), and
conflict (changes are mutually incompatible).
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class NodeType(str, enum.Enum):
    DOCUMENTATION_FRAGMENT = "documentation_fragment"
    TEMPLATE_COMPONENT = "template_component"
    SEMANTIC_CONCEPT = "semantic_concept"


class EdgeType(str, enum.Enum):
    DEPENDENCY = "dependency"
    GENERATION = "generation"
    SUPERSESSION = "supersession"
    CONFLICT = "conflict"


@dataclass
class GraphNode:
    node_type: NodeType
    properties: dict[str, Any] = field(default_factory=dict)
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attribution: str | None = None


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attribution: str | None = None

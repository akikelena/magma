"""Template component model tracked by the Knowledge Repository."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class TemplateComponent:
    """A generator template file tracked as a node in the knowledge graph."""

    path: str
    language: str
    content: str
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generation_rules: list[str] = field(default_factory=list)
    output_formats: list[str] = field(default_factory=list)

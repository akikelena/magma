"""Data models produced by the Documentation Processing Agent."""
from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class FragmentLabel(str, enum.Enum):
    """Semantic label assigned to a documentation fragment."""

    API_CONSTRAINT = "api_constraint"
    MIGRATION_INSTRUCTION = "migration_instruction"
    DEPRECATION_NOTICE = "deprecation_notice"
    ERROR_HANDLING_REQUIREMENT = "error_handling_requirement"
    UNCLASSIFIED = "unclassified"


def stable_fragment_id(source_url: str, heading_path: list[str], ordinal: int = 0) -> str:
    """Deterministic identity for "the same logical section" across
    documentation versions.

    Fragment identity must be stable across versions (not re-randomized on
    every parse) so that graph edges linking a template to the fragment(s)
    it generates from -- created once, when a template is first registered
    -- keep pointing at the right node as later versions overwrite its
    content. Identity is derived from source + heading path + `ordinal`
    (not content, since content is exactly what is expected to change
    between versions). `ordinal` disambiguates multiple fragments that
    share one heading path (e.g. several paragraphs under one heading);
    the Documentation Processing Agent assigns it as a stable per-heading
    sequence number, which holds as long as section order is preserved
    across versions -- true for the moderate edits this agent processes.
    """
    key = source_url + "\x1f" + "\x1f".join(heading_path) + "\x1f" + str(ordinal)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


@dataclass
class DocumentationFragment:
    """A single normalized, labeled unit of vendor documentation."""

    content: str
    label: FragmentLabel
    source_url: str
    vendor: str
    version: str
    heading_path: list[str] = field(default_factory=list)
    fragment_id: str = ""
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    is_code_block: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.fragment_id:
            self.fragment_id = stable_fragment_id(self.source_url, self.heading_path)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["label"] = self.label.value
        d["created_at"] = self.created_at.isoformat()
        return d

"""Change classification models produced by the Change Detection Agent.

The six change types are distinguished by two underlying attributes rather
than surface form: `intent_shift` (magnitude of semantic divergence, 0-1)
and `compatibility_preservation` (whether prior generated code remains
valid). Types are ordered here by ascending severity so that comparisons
(`ChangeType.MAJOR > ChangeType.MINOR`) read naturally; the actual
per-category attribute ranges are calibrated empirically and live in
`config/agents.yaml`.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


class ChangeType(enum.IntEnum):
    """Six semantic change categories, ascending severity."""

    EDITORIAL = 0
    PATCH = 1
    MINOR = 2
    MAJOR = 3
    DEPRECATION = 4
    BREAKING = 5


@dataclass
class DetectedChange:
    """A single classified change between two documentation versions."""

    fragment_id: str
    change_type: ChangeType
    rationale: str
    intent_shift: float
    compatibility_preserved: bool
    confidence: float
    edit_script: list[str] = field(default_factory=list)
    affected_entities: list[str] = field(default_factory=list)
    change_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_edge_case: bool = False
    # Which EscalationReason (by value) marked this change as an edge case,
    # e.g. "low_confidence" or "security_critical" -- lets the Change
    # Detection Agent report the *specific* reason at the changeset level
    # rather than collapsing every edge case into one generic flag.
    escalation_reason: str | None = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ChangeSet:
    """Filtered and grouped output of the Change Detection Agent."""

    vendor: str
    old_version: str
    new_version: str
    changes: list[DetectedChange] = field(default_factory=list)
    changeset_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def has_edge_cases(self) -> bool:
        return any(c.is_edge_case for c in self.changes)

    @property
    def max_severity(self) -> ChangeType | None:
        if not self.changes:
            return None
        return max(c.change_type for c in self.changes)

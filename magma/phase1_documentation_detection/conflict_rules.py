"""Rule-based conflict detection: the deterministic half of the Version
Coherence Agent's two conflict-identification mechanisms.

Detects three deterministic patterns, each evaluated as a graph query and
yielding a binary outcome:
  1. simultaneous modification of the same template section by different
     change sets,
  2. introduction of a cycle into the dependency graph,
  3. violation of an encoded invariant.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

from magma.core.models.graph import EdgeType
from magma.knowledge_repository.graph_store import GraphStore


class ConflictPattern(str, enum.Enum):
    SIMULTANEOUS_MODIFICATION = "simultaneous_modification"
    DEPENDENCY_CYCLE = "dependency_cycle"
    INVARIANT_VIOLATION = "invariant_violation"


@dataclass
class RuleConflict:
    pattern: ConflictPattern
    node_ids: list[str]
    detail: str


@dataclass
class Invariant:
    """A named predicate over a template node's properties.

    `check(properties) -> bool` returns True when the invariant HOLDS
    (i.e. no violation). Kept as plain callables so meta-model invariants
    can be defined in `config/` and loaded without touching this module.
    """

    name: str
    check: "callable"


def detect_simultaneous_modification(
    proposed_template_ids: list[str], changeset_owner: dict[str, str]
) -> list[RuleConflict]:
    """Flag template ids that more than one change set proposes to modify."""
    owners_by_template: dict[str, set[str]] = {}
    for template_id in proposed_template_ids:
        owner = changeset_owner.get(template_id)
        if owner is None:
            continue
        owners_by_template.setdefault(template_id, set()).add(owner)
    conflicts = []
    for template_id, owners in owners_by_template.items():
        if len(owners) > 1:
            conflicts.append(
                RuleConflict(
                    pattern=ConflictPattern.SIMULTANEOUS_MODIFICATION,
                    node_ids=[template_id],
                    detail=f"template {template_id!r} modified by change sets {sorted(owners)!r}",
                )
            )
    return conflicts


def detect_dependency_cycles(
    graph: GraphStore, affected_node_ids: list[str], new_edges: list[tuple[str, str]]
) -> list[RuleConflict]:
    """Flag any proposed dependency edge that would introduce a cycle."""
    conflicts = []
    for source_id, target_id in new_edges:
        if graph.introduces_cycle(source_id, target_id, EdgeType.DEPENDENCY):
            conflicts.append(
                RuleConflict(
                    pattern=ConflictPattern.DEPENDENCY_CYCLE,
                    node_ids=[source_id, target_id],
                    detail=f"adding dependency {source_id!r} -> {target_id!r} introduces a cycle",
                )
            )
    return conflicts


def detect_invariant_violations(
    invariants: list[Invariant], node_properties: dict[str, dict]
) -> list[RuleConflict]:
    conflicts = []
    for node_id, properties in node_properties.items():
        for invariant in invariants:
            if not invariant.check(properties):
                conflicts.append(
                    RuleConflict(
                        pattern=ConflictPattern.INVARIANT_VIOLATION,
                        node_ids=[node_id],
                        detail=f"node {node_id!r} violates invariant {invariant.name!r}",
                    )
                )
    return conflicts

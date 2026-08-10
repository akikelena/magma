"""Version Coherence Agent.

Modality: hybrid -- deterministic graph traversal composed with a
machine-learning predictor. Realizes the second line of defense: impact
analysis via transitive closure over dependency edges, identifying
inconsistencies in dependency chains and supersession relationships before
template synthesis begins.

Input (pre): a classified change set and the knowledge graph.
Output (post): a topologically sequenced change set, or a set of conflict
flags raised for human resolution.

Failure handling: any flagged conflict suspends automated processing and
escalates to the Human Expert Agent with explanatory context.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from magma.core.governance.audit_log import AuditLogger
from magma.core.governance.escalation import EscalationReason
from magma.core.interfaces.agent import Agent, AgentResult, Modality
from magma.core.models.change import ChangeSet, DetectedChange
from magma.core.models.graph import EdgeType
from magma.knowledge_repository.graph_store import GraphStore
from magma.phase1_documentation_detection.conflict_classifier import (
    ConflictFeatures,
    ConflictScorer,
    HeuristicConflictScorer,
    weighted_decision,
)
from magma.phase1_documentation_detection.conflict_rules import (
    Invariant,
    RuleConflict,
    detect_dependency_cycles,
    detect_invariant_violations,
    detect_simultaneous_modification,
)


@dataclass
class CoherenceRequest:
    change_set: ChangeSet
    # template_id -> owning changeset_id, for simultaneous-modification detection
    proposed_template_owners: dict[str, str] = field(default_factory=dict)
    new_dependency_edges: list[tuple[str, str]] = field(default_factory=list)
    invariants: list[Invariant] = field(default_factory=list)
    conflict_score_threshold: float = 0.5


@dataclass
class CoherenceOutcome:
    change_set: ChangeSet
    sequenced_template_ids: list[str]
    rule_conflicts: list[RuleConflict]
    conflict_probability: float
    conflicted: bool


class VersionCoherenceAgent(Agent[CoherenceRequest, CoherenceOutcome]):
    name = "Version Coherence Agent"
    modality = Modality.HYBRID

    def __init__(
        self,
        graph_store: GraphStore,
        conflict_scorer: ConflictScorer | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        super().__init__(blackboard=None)
        self.graph = graph_store
        self.conflict_scorer = conflict_scorer or HeuristicConflictScorer()
        self.audit = audit_logger or AuditLogger()

    def _affected_template_ids(self, changes: list[DetectedChange]) -> list[str]:
        """Traverse generation/dependency edges from each changed fragment to
        find the template components it affects, then propagate through the
        transitive closure of dependency edges to surface cascading effects."""
        affected: set[str] = set()
        for change in changes:
            for edge in self.graph.edges_to(change.fragment_id, EdgeType.GENERATION):
                affected.add(edge.source_id)
        cascaded: set[str] = set()
        for template_id in affected:
            cascaded |= self.graph.reverse_transitive_closure(template_id, EdgeType.DEPENDENCY)
        return sorted(affected | cascaded)

    def process(self, data: CoherenceRequest) -> AgentResult[CoherenceOutcome]:
        affected_ids = self._affected_template_ids(data.change_set.changes)

        rule_conflicts: list[RuleConflict] = []
        rule_conflicts += detect_simultaneous_modification(affected_ids, data.proposed_template_owners)
        rule_conflicts += detect_dependency_cycles(self.graph, affected_ids, data.new_dependency_edges)
        if data.invariants:
            node_properties = {
                nid: (self.graph.get_node(nid).properties if self.graph.get_node(nid) else {})
                for nid in affected_ids
            }
            rule_conflicts += detect_invariant_violations(data.invariants, node_properties)

        max_severity = data.change_set.max_severity
        features = ConflictFeatures(
            change_type_ordinal=int(max_severity) if max_severity is not None else 0,
            affected_node_count=len(affected_ids),
            graph_distance=1 if affected_ids else 0,
            historical_co_modification_frequency=0.0,
        )
        conflict_probability = self.conflict_scorer.score(features)
        conflicted = weighted_decision(
            rule_triggered=bool(rule_conflicts),
            ml_probability=conflict_probability,
            threshold=data.conflict_score_threshold,
        )

        sequenced: list[str] = []
        if not conflicted:
            try:
                sequenced = self.graph.topological_sort(affected_ids, EdgeType.DEPENDENCY)
            except ValueError as exc:
                rule_conflicts.append(RuleConflict(pattern="dependency_cycle", node_ids=affected_ids, detail=str(exc)))
                conflicted = True

        outcome = CoherenceOutcome(
            change_set=data.change_set,
            sequenced_template_ids=sequenced,
            rule_conflicts=rule_conflicts,
            conflict_probability=conflict_probability,
            conflicted=conflicted,
        )
        self.audit.log(
            self.name, "coherence_check", episode_id=data.change_set.changeset_id,
            conflicted=conflicted, conflict_probability=conflict_probability, rule_conflict_count=len(rule_conflicts),
        )
        return AgentResult(
            agent_name=self.name, modality=self.modality, output=outcome, success=True,
            escalate=conflicted,
            escalation_reason=EscalationReason.CONFLICTING_SPECIFICATION.value if conflicted else None,
            rationale="; ".join(c.detail for c in rule_conflicts) if rule_conflicts else None,
        )

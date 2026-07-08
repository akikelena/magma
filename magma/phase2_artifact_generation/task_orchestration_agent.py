"""Task Orchestration Agent.

Modality: deterministic service; this agent is the supervisor of the
architecture. It plans the workflow through four analyses, each realized
as graph queries and rule evaluation, so mode selection is reproducible
across runs. It never holds model state and never
calls another agent directly -- it only reads the blackboard and returns a
plan; `magma.orchestration` is responsible for dispatching each stage.

Input (pre): a validated change set (post Version Coherence Agent).
Output (post): an execution mode and an ordered task plan.

Failure: a stage that fails its quality gate is returned for rework within
the three-iteration limit, after which the case is escalated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from magma.core.governance.audit_log import AuditLogger
from magma.core.interfaces.agent import Agent, AgentResult, Modality
from magma.core.models.change import ChangeSet, ChangeType
from magma.core.models.execution import (
    ExecutionMode,
    ExecutionPlan,
    GenerationMode,
    MaintenanceCategory,
    TemplateTask,
)
from magma.core.models.graph import EdgeType
from magma.knowledge_repository.graph_store import GraphStore

# Change-type decision table: maps classification label to maintenance category.
_MAINTENANCE_CATEGORY_BY_CHANGE_TYPE = {
    ChangeType.EDITORIAL: MaintenanceCategory.CORRECTIVE,
    ChangeType.PATCH: MaintenanceCategory.CORRECTIVE,
    ChangeType.MINOR: MaintenanceCategory.ADDITIVE,
    ChangeType.MAJOR: MaintenanceCategory.ADDITIVE,
    ChangeType.DEPRECATION: MaintenanceCategory.TRANSFORMATIVE,
    ChangeType.BREAKING: MaintenanceCategory.TRANSFORMATIVE,
}

_LOCALIZED_LINE_THRESHOLD = 0.30  # <30% of template lines affected -> targeted modification


@dataclass
class OrchestrationRequest:
    change_set: ChangeSet
    # Optional explicit override, mainly for tests/simple manual use. When
    # None, the agent performs its own affected-artifact + dependency
    # analysis against the graph rather than depending on another agent's
    # already-computed result -- this is what keeps the blackboard
    # architecture's "read state, not another agent's output" principle
    # true for this agent specifically.
    sequenced_template_ids: list[str] | None = None
    # template_id -> (changed_lines, total_lines), for scope analysis
    template_line_stats: dict[str, tuple[int, int]] = field(default_factory=dict)
    is_parallelizable: bool = False
    max_rework_iterations: int = 3


class TaskOrchestrationAgent(Agent[OrchestrationRequest, ExecutionPlan]):
    name = "Task Orchestration Agent"
    modality = Modality.ALGORITHMIC

    def __init__(self, graph_store: GraphStore | None = None, audit_logger: AuditLogger | None = None) -> None:
        super().__init__(blackboard=None)
        self.graph = graph_store
        self.audit = audit_logger or AuditLogger()

    # -- Affected-artifact analysis -------------------------------------------

    def _affected_template_ids(self, change_set: ChangeSet) -> list[str]:
        """Traverse GENERATION edges from each changed fragment to find the
        template files that require modification."""
        if self.graph is None:
            return []
        affected: set[str] = set()
        for change in change_set.changes:
            for edge in self.graph.edges_to(change.fragment_id, EdgeType.GENERATION):
                affected.add(edge.source_id)
        return sorted(affected)

    def _template_metadata(self, template_id: str) -> tuple[list[str], list[str], str]:
        """Pull generation rules, output formats, and architectural
        constraints straight from the template's graph-node properties.
        Missing properties default to empty, since not every template
        node is required to declare them."""
        if self.graph is None:
            return [], [], ""
        node = self.graph.get_node(template_id)
        if node is None:
            return [], [], ""
        return (
            list(node.properties.get("generation_rules", [])),
            list(node.properties.get("output_formats", [])),
            str(node.properties.get("architectural_constraints", "")),
        )

    # -- Dependency analysis ----------------------------------------------------

    def _sequence_templates(self, template_ids: list[str]) -> list[str]:
        """Builds the DAG of template dependencies over `template_ids` and
        fixes the modification order by topological sort. Also propagates
        through the transitive closure of dependency edges
        so cascading dependents are included, mirroring the Version
        Coherence Agent's impact analysis but computed independently here
        for workflow-planning purposes rather than conflict detection."""
        if self.graph is None:
            return template_ids
        cascaded: set[str] = set(template_ids)
        for template_id in template_ids:
            cascaded |= self.graph.reverse_transitive_closure(template_id, EdgeType.DEPENDENCY)
        try:
            return self.graph.topological_sort(sorted(cascaded), EdgeType.DEPENDENCY)
        except ValueError:
            # A cycle here should already have been caught by the Version
            # Coherence Agent's rule-based conflict check upstream; fall
            # back to a stable order rather than raising mid-plan.
            return sorted(cascaded)

    def _change_scope(self, template_ids: list[str]) -> str:
        """Localized vs distributed, by affected node/edge count."""
        return "localized" if len(template_ids) <= 1 else "distributed"

    def _maintenance_category(self, change_set: ChangeSet) -> MaintenanceCategory:
        severity = change_set.max_severity
        if severity is None:
            return MaintenanceCategory.ADDITIVE
        return _MAINTENANCE_CATEGORY_BY_CHANGE_TYPE[severity]

    def _generation_mode(self, template_id: str, line_stats: dict[str, tuple[int, int]]) -> GenerationMode:
        changed, total = line_stats.get(template_id, (0, 0))
        if total == 0:
            return GenerationMode.TARGETED_MODIFICATION
        fraction = changed / total
        return (
            GenerationMode.TARGETED_MODIFICATION
            if fraction < _LOCALIZED_LINE_THRESHOLD
            else GenerationMode.COMPREHENSIVE_SYNTHESIS
        )

    def _select_execution_mode(self, request: OrchestrationRequest, scope: str) -> ExecutionMode:
        if request.is_parallelizable:
            return ExecutionMode.PARALLEL
        return ExecutionMode.LIGHTWEIGHT if scope == "localized" else ExecutionMode.COMPREHENSIVE

    def process(self, data: OrchestrationRequest) -> AgentResult[ExecutionPlan]:
        if data.sequenced_template_ids is not None:
            sequenced_ids = data.sequenced_template_ids
        else:
            affected = self._affected_template_ids(data.change_set)
            sequenced_ids = self._sequence_templates(affected)

        scope = self._change_scope(sequenced_ids)
        maintenance_category = self._maintenance_category(data.change_set)
        execution_mode = self._select_execution_mode(data, scope)

        tasks = []
        for template_id in sequenced_ids:
            generation_rules, output_formats, architectural_constraints = self._template_metadata(template_id)
            tasks.append(
                TemplateTask(
                    template_id=template_id,
                    maintenance_category=maintenance_category,
                    generation_mode=self._generation_mode(template_id, data.template_line_stats),
                    instructions=(
                        f"Apply {maintenance_category.value} changes from changeset "
                        f"{data.change_set.changeset_id} to {template_id}."
                    ),
                    generation_rules=generation_rules,
                    output_formats=output_formats,
                    architectural_constraints=architectural_constraints,
                )
            )

        plan = ExecutionPlan(
            changeset_id=data.change_set.changeset_id, execution_mode=execution_mode, ordered_tasks=tasks
        )
        self.audit.log(
            self.name, "plan_workflow", episode_id=data.change_set.changeset_id,
            execution_mode=execution_mode.value, task_count=len(tasks), scope=scope,
        )
        return AgentResult(agent_name=self.name, modality=self.modality, output=plan, success=True)

    def evaluate_quality_gate(self, *, attempt: int, passed: bool, max_iterations: int | None = None) -> bool:
        """True if the stage should be reworked (retried), False if it
        must instead be escalated to the Human Expert Agent."""
        limit = max_iterations if max_iterations is not None else 3
        return not passed and attempt < limit

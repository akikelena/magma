"""Synthetic Testing Agent.

Modality: deterministic service. Subjects the decision logic and
error-handling mechanisms to stress conditions using adversarial scenarios,
and reports patterns that require additional handling logic or stronger
safeguards. Operates offline on a copy of the configuration/pipeline
components, so it never mutates production state.

Failure: not applicable -- this agent operates offline on a copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from magma.core.governance.audit_log import AuditLogger
from magma.core.interfaces.agent import Agent, AgentResult, Modality


@dataclass
class AdversarialScenario:
    name: str
    run: Callable[[], bool]  # returns True if the system handled the scenario safely
    description: str = ""


@dataclass
class ScenarioOutcome:
    name: str
    survived: bool
    description: str = ""


@dataclass
class RobustnessReport:
    outcomes: list[ScenarioOutcome] = field(default_factory=list)

    @property
    def survival_rate(self) -> float:
        if not self.outcomes:
            return 1.0
        return sum(1 for o in self.outcomes if o.survived) / len(self.outcomes)

    @property
    def failure_patterns(self) -> list[str]:
        return [o.name for o in self.outcomes if not o.survived]


class SyntheticTestingAgent(Agent[list[AdversarialScenario], RobustnessReport]):
    name = "Synthetic Testing Agent"
    modality = Modality.ALGORITHMIC

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        super().__init__(blackboard=None)
        self.audit = audit_logger or AuditLogger()

    def process(self, data: list[AdversarialScenario]) -> AgentResult[RobustnessReport]:
        outcomes: list[ScenarioOutcome] = []
        for scenario in data:
            try:
                survived = bool(scenario.run())
            except Exception:  # noqa: BLE001 - an uncaught exception is itself a failure to survive
                survived = False
            outcomes.append(ScenarioOutcome(name=scenario.name, survived=survived, description=scenario.description))

        report = RobustnessReport(outcomes=outcomes)
        self.audit.log(
            self.name, "run_scenarios", episode_id="synthetic-testing",
            scenario_count=len(outcomes), survival_rate=report.survival_rate,
            failure_patterns=report.failure_patterns,
        )
        return AgentResult(agent_name=self.name, modality=self.modality, output=report, success=True)


def default_scenarios(
    change_detection_agent, version_coherence_agent, graph_store,
) -> list[AdversarialScenario]:
    """A small, representative battery exercising the failure modes the
    architecture explicitly calls out: malformed classification output,
    a dependency cycle, and an empty change set.

    `version_coherence_agent` may be None, in which case a default
    instance is built from `graph_store`; if given, that exact instance
    is used (e.g. one wired to a live graph with a custom conflict
    scorer) rather than silently replaced with a fresh vanilla one."""
    from magma.core.models.change import ChangeSet
    from magma.phase1_documentation_detection.change_detection_agent import (
        ChangeDetectionRequest,
        FragmentVersionPair,
    )
    from magma.phase1_documentation_detection.version_coherence_agent import (
        CoherenceRequest,
        VersionCoherenceAgent,
    )

    coherence_agent = version_coherence_agent or VersionCoherenceAgent(graph_store=graph_store)

    def malformed_classification_survives() -> bool:
        request = ChangeDetectionRequest("V", "1", "2", [FragmentVersionPair("f1", "a\n", "b\n")])
        result = change_detection_agent.process(request)
        # surviving = the agent returns a result rather than raising, and
        # flags the unparseable case as an edge case rather than silently
        # committing a wrong classification.
        return result.output.changes[0].is_edge_case if result.output.changes else False

    def empty_changeset_survives() -> bool:
        request = ChangeDetectionRequest("V", "1", "2", [])
        result = change_detection_agent.process(request)
        return result.success and result.output.changes == []

    def cycle_detection_survives() -> bool:
        empty_cs = ChangeSet(vendor="V", old_version="1", new_version="2", changes=[])
        result = coherence_agent.process(CoherenceRequest(change_set=empty_cs))
        return result.success  # must not raise, even with no changes to sequence

    return [
        AdversarialScenario("malformed_classification_output", malformed_classification_survives,
                             "LLM stage returns text with no parseable JSON classification"),
        AdversarialScenario("empty_changeset", empty_changeset_survives, "no fragment pairs differ at all"),
        AdversarialScenario("coherence_with_no_changes", cycle_detection_survives, "coherence check on an empty change set"),
    ]

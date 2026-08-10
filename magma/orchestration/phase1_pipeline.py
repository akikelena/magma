"""Wires the four Documentation Change Detection Module agents together
end-to-end: Documentation Processing -> Knowledge Repository -> Change
Detection -> Version Coherence.

This is the concrete realization of "the four agents acquire and structure
documentation, persist it in a versioned knowledge base, classify what has
changed, and confirm that the proposed updates are mutually consistent and
correctly ordered." Agents never call each other directly; this pipeline
plays the narrow role the Task Orchestration Agent will later play for the
full four-module workflow, but scoped to Phase 1 alone so it can be
exercised (and tested) independently of Phases 2-4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from magma.core.governance.audit_log import AuditLogger
from magma.core.governance.escalation import EscalationPolicy, EscalationReason
from magma.core.interfaces.llm_provider import LLMProvider
from magma.core.models.change import ChangeSet
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.phase3_validation.human_expert_agent import HumanExpertAgent, ReviewTicket
from magma.phase1_documentation_detection.change_detection_agent import (
    ChangeDetectionAgent,
    ChangeDetectionRequest,
    FragmentVersionPair,
)
from magma.phase1_documentation_detection.conflict_classifier import ConflictScorer
from magma.phase1_documentation_detection.documentation_processing_agent import (
    DocumentationProcessingAgent,
    RawDocument,
)
from magma.phase1_documentation_detection.version_coherence_agent import (
    CoherenceOutcome,
    CoherenceRequest,
    VersionCoherenceAgent,
)


@dataclass
class Phase1Result:
    change_set: ChangeSet
    coherence: CoherenceOutcome
    requires_human_review: bool
    escalation_reasons: list[str] = field(default_factory=list)
    review_tickets: list[ReviewTicket] = field(default_factory=list)


def _align_fragments_by_heading(old_fragments, new_fragments) -> list[FragmentVersionPair]:
    """Match fragments across versions by heading path so the diff engine
    compares like with like. A heading present only in one version is
    diffed against an empty string, which the Myers-diff preprocessor
    correctly reports as a full addition/removal rather than "identical"."""
    old_by_heading = {tuple(f.heading_path): f for f in old_fragments if not f.is_code_block or f.heading_path}
    new_by_heading = {tuple(f.heading_path): f for f in new_fragments if not f.is_code_block or f.heading_path}
    all_headings = sorted(set(old_by_heading) | set(new_by_heading), key=lambda h: (len(h), h))

    pairs = []
    for heading in all_headings:
        old_fragment = old_by_heading.get(heading)
        new_fragment = new_by_heading.get(heading)
        fragment_id = (new_fragment or old_fragment).fragment_id
        pairs.append(
            FragmentVersionPair(
                fragment_id=fragment_id,
                old_text=old_fragment.content if old_fragment else "",
                new_text=new_fragment.content if new_fragment else "",
            )
        )
    return pairs


class Phase1Pipeline:
    """Runs the Documentation Change Detection Module end to end."""

    def __init__(
        self,
        knowledge_repository: KnowledgeRepositoryAgent,
        llm_provider: LLMProvider,
        confidence_threshold: float = 0.70,
        conflict_scorer: ConflictScorer | None = None,
        human_expert_agent: HumanExpertAgent | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.audit = audit_logger or AuditLogger()
        self.knowledge_repository = knowledge_repository
        # Shared across modules by `MagmaWorkflow` so every escalation --
        # not just Phase 3's -- lands in one review queue.
        self.human_expert_agent = human_expert_agent or HumanExpertAgent(audit_logger=self.audit)
        self.documentation_processing_agent = DocumentationProcessingAgent(
            knowledge_repository=knowledge_repository, audit_logger=self.audit
        )
        self.change_detection_agent = ChangeDetectionAgent(
            llm_provider=llm_provider,
            confidence_threshold=confidence_threshold,
            escalation_policy=EscalationPolicy(confidence_threshold=confidence_threshold),
            audit_logger=self.audit,
        )
        self.version_coherence_agent = VersionCoherenceAgent(
            graph_store=knowledge_repository.graph, conflict_scorer=conflict_scorer, audit_logger=self.audit
        )

    def run(self, old_document: RawDocument, new_document: RawDocument) -> Phase1Result:
        old_result = self.documentation_processing_agent.process(old_document)
        new_result = self.documentation_processing_agent.process(new_document)

        old_fragments = old_result.output or []
        new_fragments = new_result.output or []
        pairs = _align_fragments_by_heading(old_fragments, new_fragments)

        detection_request = ChangeDetectionRequest(
            vendor=new_document.vendor,
            old_version=old_document.version,
            new_version=new_document.version,
            fragment_pairs=pairs,
        )
        detection_result = self.change_detection_agent.process(detection_request)
        change_set = detection_result.output

        coherence_result = self.version_coherence_agent.process(CoherenceRequest(change_set=change_set))
        coherence_outcome = coherence_result.output

        escalation_reasons = []
        review_tickets: list[ReviewTicket] = []
        if detection_result.escalate and detection_result.escalation_reason:
            escalation_reasons.append(detection_result.escalation_reason)
            review_tickets.append(self.human_expert_agent.enqueue(
                episode_id=change_set.changeset_id,
                reason=EscalationReason(detection_result.escalation_reason),
                context={"stage": "change_detection", "rationale": detection_result.rationale},
            ))
        if coherence_result.escalate and coherence_result.escalation_reason:
            escalation_reasons.append(coherence_result.escalation_reason)
            review_tickets.append(self.human_expert_agent.enqueue(
                episode_id=change_set.changeset_id,
                reason=EscalationReason(coherence_result.escalation_reason),
                context={"stage": "version_coherence", "rationale": coherence_result.rationale},
            ))

        self.audit.log(
            "Phase1Pipeline", "run_complete", episode_id=change_set.changeset_id,
            requires_human_review=bool(escalation_reasons), escalation_reasons=escalation_reasons,
        )

        return Phase1Result(
            change_set=change_set,
            coherence=coherence_outcome,
            requires_human_review=bool(escalation_reasons),
            escalation_reasons=escalation_reasons,
            review_tickets=review_tickets,
        )

"""Change Detection Agent.

Modality: hybrid -- deterministic first stage (Myers diff) composed with an
LLM second stage (chain-of-thought classification), so determinism and
flexibility are separated by stage rather than dissolved within one unit.

Input (pre): two documentation versions for the same artifact.
Output (post): a filtered and grouped `ChangeSet`, each change carrying one
of six severity labels and a rationale.

Failure handling: changes whose classification confidence falls below
`confidence_threshold` are labeled edge cases (`is_edge_case=True`) and
routed to human review rather than forced into a category -- realizing the
"escalate rather than commit" rule for the first line of defense.
"""
from __future__ import annotations

from dataclasses import dataclass

from magma.core.governance.audit_log import AuditLogger
from magma.core.utils.timing import timed
from magma.core.governance.escalation import EscalationPolicy, EscalationReason
from magma.core.interfaces.agent import Agent, AgentResult, Modality
from magma.core.interfaces.llm_provider import LLMProvider
from magma.core.models.change import ChangeSet, ChangeType, DetectedChange
from magma.phase1_documentation_detection.change_classification_prompt import (
    SYSTEM_PROMPT,
    ClassificationParseError,
    build_prompt,
    parse_classification,
)
from magma.phase1_documentation_detection.diff_engine import compute_diff, is_editorial_only


@dataclass
class FragmentVersionPair:
    fragment_id: str
    old_text: str
    new_text: str


@dataclass
class ChangeDetectionRequest:
    vendor: str
    old_version: str
    new_version: str
    fragment_pairs: list[FragmentVersionPair]


class ChangeDetectionAgent(Agent[ChangeDetectionRequest, ChangeSet]):
    name = "Change Detection Agent"
    modality = Modality.HYBRID

    def __init__(
        self,
        llm_provider: LLMProvider,
        confidence_threshold: float = 0.70,
        temperature: float = 0.0,
        seed: int | None = 42,
        escalation_policy: EscalationPolicy | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        super().__init__(blackboard=None)
        self.llm = llm_provider
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.seed = seed
        self.escalation_policy = escalation_policy or EscalationPolicy(confidence_threshold=confidence_threshold)
        self.audit = audit_logger or AuditLogger()

    def process(self, data: ChangeDetectionRequest) -> AgentResult[ChangeSet]:
        changes: list[DetectedChange] = []
        edge_case_count = 0
        total_llm_ms = 0.0

        for pair in data.fragment_pairs:
            diff = compute_diff(pair.old_text, pair.new_text)

            # First line of defense: deterministic preprocessing removes
            # non-semantic changes before the (costly) LLM stage runs.
            if diff.is_identical:
                continue
            if is_editorial_only(diff):
                changes.append(
                    DetectedChange(
                        fragment_id=pair.fragment_id,
                        change_type=ChangeType.EDITORIAL,
                        rationale="whitespace-only diff filtered by deterministic preprocessor",
                        intent_shift=0.0,
                        compatibility_preserved=True,
                        confidence=1.0,
                        edit_script=diff.edit_script,
                    )
                )
                continue

            prompt = build_prompt(pair.fragment_id, diff.edit_script)
            with timed() as elapsed:
                response = self.llm.complete(
                    prompt, system=SYSTEM_PROMPT, temperature=self.temperature, max_tokens=1024, seed=self.seed
                )
            total_llm_ms += elapsed.ms

            try:
                parsed = parse_classification(response.text)
            except ClassificationParseError as exc:
                self.audit.log(self.name, "classification_parse_failure", episode_id=pair.fragment_id, error=str(exc))
                # confidence=0.0 is the least trustworthy classification
                # possible, so this must carry the same LOW_CONFIDENCE
                # escalation_reason a normally-parsed low-confidence
                # result would get -- omitting it left edge_case_count
                # incremented (escalate=True below) but change_reasons
                # empty, so Phase1Result.requires_human_review silently
                # stayed False for exactly the case that most needs review.
                changes.append(
                    DetectedChange(
                        fragment_id=pair.fragment_id,
                        change_type=ChangeType.MAJOR,
                        rationale=f"unparseable LLM output, defaulting to edge case: {exc}",
                        intent_shift=0.5,
                        compatibility_preserved=False,
                        confidence=0.0,
                        edit_script=diff.edit_script,
                        is_edge_case=True,
                        escalation_reason=EscalationReason.LOW_CONFIDENCE.value,
                    )
                )
                edge_case_count += 1
                continue

            # Security-critical trigger checked ahead of the confidence
            # check: security-sensitive behavior (authentication,
            # encryption) is its own escalation condition, independent of
            # how confident the classifier was.
            reason = self.escalation_policy.evaluate(
                affected_text=pair.old_text + " " + pair.new_text + " " + parsed.rationale
            )
            if reason is None:
                reason = self.escalation_policy.evaluate(confidence=parsed.confidence)
            is_edge_case = reason is not None
            if is_edge_case:
                edge_case_count += 1

            changes.append(
                DetectedChange(
                    fragment_id=pair.fragment_id,
                    change_type=parsed.change_type,
                    rationale=parsed.rationale,
                    intent_shift=parsed.intent_shift,
                    compatibility_preserved=parsed.compatibility_preserved,
                    confidence=parsed.confidence,
                    edit_script=diff.edit_script,
                    affected_entities=parsed.affected_entities,
                    is_edge_case=is_edge_case,
                    escalation_reason=reason.value if reason else None,
                )
            )

        change_set = ChangeSet(vendor=data.vendor, old_version=data.old_version, new_version=data.new_version, changes=changes)

        # Report the most specific escalation reason across all changes,
        # preferring security-critical over low-confidence since it is the
        # hardest requirement among the escalation triggers.
        change_reasons = {c.escalation_reason for c in changes if c.escalation_reason}
        if EscalationReason.SECURITY_CRITICAL.value in change_reasons:
            overall_reason = EscalationReason.SECURITY_CRITICAL.value
        elif change_reasons:
            overall_reason = EscalationReason.LOW_CONFIDENCE.value
        else:
            overall_reason = None

        self.audit.log(
            self.name, "detect_changes", episode_id=change_set.changeset_id,
            change_count=len(changes), edge_case_count=edge_case_count, duration_ms=total_llm_ms,
        )
        return AgentResult(
            agent_name=self.name, modality=self.modality, output=change_set, success=True,
            escalate=edge_case_count > 0,
            escalation_reason=overall_reason,
        )

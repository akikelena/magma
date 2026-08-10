"""Explainability Agent.

Modality: autonomous LLM agent. Realizes the fifth line of defense:
explanation and audit, producing a traceable rationale record for each
maintenance episode that enables post-hoc identification of latent errors
that escaped the preceding four lines.

Input (pre): an episode record (the audit trail) and its validation
outcomes.
Output (post): a structured rationale and audit-trail record.

Failure: an episode lacking sufficient record is logged as incompletely
explainable rather than given a fabricated rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from magma.core.governance.audit_log import AuditLogger, AuditRecord
from magma.core.utils.timing import timed
from magma.core.interfaces.agent import Agent, AgentResult, Modality
from magma.core.interfaces.llm_provider import LLMProvider

_SYSTEM_PROMPT = (
    "You write a short, human-readable rationale for one MAGMA maintenance episode, given its "
    "audit trail. State: why the modification was required, which templates/artifacts it "
    "affected, your confidence given the validation outcomes, and the reasoning path from "
    "documentation change to deployed generator. Be concrete and brief (5-8 sentences)."
)

_MINIMUM_RECORDS_FOR_EXPLANATION = 1


@dataclass
class ExplanationRequest:
    episode_id: str
    audit_records: list[AuditRecord]
    validation_passed: bool | None = None


@dataclass
class ExplanationRecord:
    episode_id: str
    narrative: str
    confidence: float
    reasoning_path: list[str] = field(default_factory=list)
    is_complete: bool = True


class ExplainabilityAgent(Agent[ExplanationRequest, ExplanationRecord]):
    name = "Explainability Agent"
    modality = Modality.LLM

    def __init__(self, llm_provider: LLMProvider, audit_logger: AuditLogger | None = None) -> None:
        super().__init__(blackboard=None)
        self.llm = llm_provider
        self.audit = audit_logger or AuditLogger()

    def process(self, data: ExplanationRequest) -> AgentResult[ExplanationRecord]:
        if len(data.audit_records) < _MINIMUM_RECORDS_FOR_EXPLANATION:
            record = ExplanationRecord(
                episode_id=data.episode_id, narrative="", confidence=0.0,
                reasoning_path=[], is_complete=False,
            )
            self.audit.log(self.name, "incomplete_explanation", episode_id=data.episode_id)
            return AgentResult(
                agent_name=self.name, modality=self.modality, output=record, success=False,
                rationale="insufficient audit trail to construct a rationale",
            )

        reasoning_path = [f"{r.agent_name}: {r.action}" for r in data.audit_records]
        trail_text = "\n".join(reasoning_path)
        prompt = (
            f"Episode: {data.episode_id}\nValidation passed: {data.validation_passed}\n\nAudit trail:\n{trail_text}"
        )
        with timed() as elapsed:
            response = self.llm.complete(prompt, system=_SYSTEM_PROMPT, temperature=0.0)

        confidence = 0.9 if data.validation_passed else 0.5 if data.validation_passed is not None else 0.7
        record = ExplanationRecord(
            episode_id=data.episode_id, narrative=response.text, confidence=confidence,
            reasoning_path=reasoning_path, is_complete=True,
        )
        self.audit.log(
            self.name, "explain_episode", episode_id=data.episode_id,
            record_count=len(data.audit_records), duration_ms=elapsed.ms,
        )
        return AgentResult(agent_name=self.name, modality=self.modality, output=record, success=True)

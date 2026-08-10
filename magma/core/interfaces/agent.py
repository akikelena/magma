"""Core Agent interface shared by every MAGMA component.

Every component is assigned exactly one of four processing modalities:

* ALGORITHMIC   -- a deterministic service whose behavior is fixed in code.
* LLM           -- an autonomous agent that directs its own reasoning/tool use.
* HYBRID        -- a deterministic stage composed with an LLM/ML stage behind
                    an explicit interface.
* HUMAN         -- a human-in-the-loop component.

Components never address one another directly. Each agent reads its inputs
from the blackboard (the Knowledge Repository) and writes its outputs back
to it; the Task Orchestration Agent is the only component that decides what
runs next. Concretely, this means `Agent.process` takes and returns plain
data (or blackboard references), never a handle to another agent.
"""
from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Modality(str, enum.Enum):
    """The four processing modalities a MAGMA component can be assigned."""

    ALGORITHMIC = "algorithmic"
    LLM = "llm"
    HYBRID = "hybrid"
    HUMAN = "human"


@dataclass(frozen=True)
class AgentResult(Generic[OutputT]):
    """Envelope returned by every agent invocation.

    Wrapping the raw output lets the Task Orchestration Agent make routing
    decisions (retry, escalate, advance) without inspecting agent-specific
    payloads, and gives the Explainability Agent a uniform record to read.
    """

    agent_name: str
    modality: Modality
    output: OutputT | None
    success: bool
    confidence: float | None = None
    escalate: bool = False
    escalation_reason: str | None = None
    rationale: str | None = None
    produced_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent(abc.ABC, Generic[InputT, OutputT]):
    """Base class for every MAGMA component.

    Subclasses declare their modality and implement `process`. `process`
    must be a pure function of its input plus whatever it reads from the
    blackboard -- it must not call other agents directly.
    """

    #: Human-readable component name, e.g. "Documentation Processing Agent".
    name: str = "Agent"
    #: One of the four processing modalities.
    modality: Modality = Modality.ALGORITHMIC

    def __init__(self, blackboard: Any | None = None) -> None:
        self.blackboard = blackboard

    @abc.abstractmethod
    def process(self, data: InputT) -> AgentResult[OutputT]:
        """Execute this agent's single responsibility and return a result.

        Implementations should catch and classify their own failures rather
        than raising, so the caller (Task Orchestration Agent) can apply a
        uniform retry/escalate policy across heterogeneous agents.
        """
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        """Return the agent-card summary (modality + name) for audit logs."""
        return {"name": self.name, "modality": self.modality.value}

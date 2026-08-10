"""LLM abstraction interface.

Every autonomous LLM agent (Prompt Generation, Code Generation,
Explainability, and the LLM stage of Change Detection) calls models through
this interface rather than a vendor SDK directly, so the concrete provider
can be swapped (see `magma.llm_providers`) without touching agent logic.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider(abc.ABC):
    """Abstraction over a chat-completion LLM backend."""

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> LLMResponse:
        """Return a single completion for `prompt`."""
        raise NotImplementedError

    @abc.abstractmethod
    def embed(self, texts: list[str], *, dimensions: int | None = None) -> list[list[float]]:
        """Return embedding vectors for `texts`, optionally truncated to
        `dimensions` (Matryoshka Representation Learning truncation)."""
        raise NotImplementedError

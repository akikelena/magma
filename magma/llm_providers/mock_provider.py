"""Deterministic mock LLM provider for tests, CI, and offline development.

Accepts an optional `responder` callable (`prompt -> str`) so unit tests can
script exact outputs (e.g. a canned JSON change classification) without any
network access or API key.
"""
from __future__ import annotations

from typing import Callable

from magma.core.interfaces.llm_provider import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    def __init__(self, responder: Callable[[str], str] | None = None) -> None:
        self._responder = responder or (lambda prompt: "")
        self.calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> LLMResponse:
        self.calls.append(prompt)
        text = self._responder(prompt)
        return LLMResponse(text=text, model="mock-echo", prompt_tokens=len(prompt.split()), completion_tokens=len(text.split()))

    def embed(self, texts: list[str], *, dimensions: int | None = None) -> list[list[float]]:
        dims = dimensions or 8
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text) or 1
            vectors.append([((seed * (i + 1)) % 97) / 97.0 for i in range(dims)])
        return vectors

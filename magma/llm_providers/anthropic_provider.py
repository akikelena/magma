"""Anthropic (Claude) implementation of `LLMProvider`.

LLM-based agents use Claude at temperature 0.0 with a fixed seed to reduce
stochastic variation at decision points, supporting reproducible evaluation
across independent runs. The `anthropic` SDK is imported lazily so this
module can be imported without the package installed or an API key set;
only instantiating `AnthropicProvider` requires them.
"""
from __future__ import annotations

from magma.core.interfaces.llm_provider import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5") -> None:
        import anthropic  # noqa: F401

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> LLMResponse:
        kwargs = {}
        if system is not None:
            kwargs["system"] = system
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def embed(self, texts: list[str], *, dimensions: int | None = None) -> list[list[float]]:
        raise NotImplementedError(
            "AnthropicProvider does not serve embeddings; use "
            "magma.knowledge_repository.embeddings.OpenAIEmbeddingBackend"
        )

"""Fine-tuning dataset assembly: fine-tunes the generation models on the
validated outputs of successful episodes.

`FineTuningBackend` is the pluggable dispatch interface -- this module
owns collecting and shaping the training data (prompt/completion pairs
from episodes whose artifact passed Phase 3 validation); actually
submitting a fine-tuning job is provider-specific and deliberately kept
outside `magma.llm_providers`, since not every provider used in
`config/llm_models.yaml` need support it. `NoOpFineTuningBackend` is the
default: it validates and logs the dataset without dispatching anything,
so this is safe to call with no fine-tuning provider configured.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainingExample:
    episode_id: str
    template_id: str
    prompt: str
    completion: str


class FineTuningBackend(abc.ABC):
    @abc.abstractmethod
    def submit(self, examples: list[TrainingExample]) -> str:
        """Dispatch a fine-tuning job over `examples`, returning a job id
        (or other backend-specific handle)."""
        raise NotImplementedError


class NoOpFineTuningBackend(FineTuningBackend):
    """Default backend: validates and records the dataset without
    submitting a real fine-tuning job. Production deployments swap this
    for a provider-specific backend (e.g. wrapping the target LLM
    provider's fine-tuning API) without touching the collection logic in
    `FineTuningDatasetCollector`."""

    def __init__(self) -> None:
        self.submitted_batches: list[list[TrainingExample]] = []

    def submit(self, examples: list[TrainingExample]) -> str:
        self.submitted_batches.append(list(examples))
        return f"noop-batch-{len(self.submitted_batches)}-of-{len(examples)}-examples"


@dataclass
class FineTuningDatasetCollector:
    """Accumulates training examples from successful episodes and hands
    them to a `FineTuningBackend` when a fine-tuning round is triggered."""

    backend: FineTuningBackend = field(default_factory=NoOpFineTuningBackend)
    _pending: list[TrainingExample] = field(default_factory=list)

    def record_episode(self, *, episode_id: str, template_id: str, prompt: str, completion: str, success: bool) -> None:
        """Only successful episodes -- artifacts that passed Phase 3
        validation -- are eligible training data."""
        if not success:
            return
        self._pending.append(TrainingExample(episode_id=episode_id, template_id=template_id, prompt=prompt, completion=completion))

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def dispatch(self, *, min_batch_size: int = 1) -> str | None:
        """Submit accumulated examples to the backend, if there are enough
        to be worth a round. Returns the backend's job handle, or None if
        the batch was below `min_batch_size`."""
        if len(self._pending) < min_batch_size:
            return None
        job_id = self.backend.submit(self._pending)
        self._pending = []
        return job_id

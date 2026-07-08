"""Validation outcome models for the Automated Validation Agent (Phase 3)."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class ValidationLayer(enum.IntEnum):
    """Six layers, ordered by ascending cost so cheaper checks run first."""

    SYNTAX = 0
    STYLE = 1
    FUNCTIONAL = 2
    INTEGRATION = 3
    SECURITY = 4
    PERFORMANCE = 5


@dataclass
class LayerResult:
    layer: ValidationLayer
    passed: bool
    details: str = ""
    diagnostics: dict[str, str] = field(default_factory=dict)


@dataclass
class ValidationOutcome:
    layer_results: list[LayerResult] = field(default_factory=list)
    attempt: int = 1
    max_attempts: int = 3

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.layer_results)

    @property
    def first_failure(self) -> LayerResult | None:
        for r in self.layer_results:
            if not r.passed:
                return r
        return None

    @property
    def exhausted_retries(self) -> bool:
        return not self.passed and self.attempt >= self.max_attempts

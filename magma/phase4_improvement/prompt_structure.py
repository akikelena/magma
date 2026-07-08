"""Supervised prompt-structure prediction: predicts effective prompt
structures from change characteristics, where the training set pairs
change-feature vectors with the prompt configurations that led to
successful updates.

A "prompt structure" here is which optional context sections the Prompt
Generation Agent includes (see `magma.phase2_artifact_generation.
prompt_generation_agent.PromptGenerationRequest`): a minimal prompt for
small, low-risk changes, a standard prompt for typical ones, and a rich
prompt (larger token budget, more exemplars) for high-severity or
security-related changes where extra context measurably helps.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class PromptStructureLabel(str, enum.Enum):
    MINIMAL = "minimal"     # small token budget, no exemplars
    STANDARD = "standard"   # default budget, a couple of exemplars
    RICH = "rich"           # large budget, max exemplars, all metadata sections


@dataclass(frozen=True)
class PromptStructureFeatures:
    change_type_ordinal: int   # ChangeType, ascending severity
    affected_entity_count: int
    is_security_related: bool


@dataclass(frozen=True)
class PromptStructureExample:
    features: PromptStructureFeatures
    label: PromptStructureLabel
    success: bool


def _to_vector(features: PromptStructureFeatures) -> list[float]:
    return [
        features.change_type_ordinal / 5.0,
        min(features.affected_entity_count / 10.0, 1.0),
        1.0 if features.is_security_related else 0.0,
    ]


def _heuristic_predict(features: PromptStructureFeatures) -> PromptStructureLabel:
    if features.is_security_related or features.change_type_ordinal >= 4:
        return PromptStructureLabel.RICH
    if features.change_type_ordinal >= 2 or features.affected_entity_count > 2:
        return PromptStructureLabel.STANDARD
    return PromptStructureLabel.MINIMAL


class PromptStructurePredictor:
    """Predicts the prompt structure most likely to lead to a successful
    episode, given a change's features.

    `scikit-learn` is imported lazily; until enough *successful* labeled
    episodes have accumulated (`fit` is a no-op below that threshold), a
    deterministic heuristic (`_heuristic_predict`) stands in, mirroring the
    fallback pattern used by `UsageFeedbackAgent.SuccessProbabilityEstimator`.
    """

    _MIN_EXAMPLES_TO_FIT = 8

    def __init__(self) -> None:
        self._model = None
        self._fitted = False

    def fit(self, examples: list[PromptStructureExample]) -> None:
        successful = [e for e in examples if e.success]
        if len(successful) < self._MIN_EXAMPLES_TO_FIT:
            return
        from sklearn.tree import DecisionTreeClassifier  # noqa: F401 (lazy import)

        X = [_to_vector(e.features) for e in successful]
        y = [e.label.value for e in successful]
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X, y)
        self._model = model
        self._fitted = True

    def predict(self, features: PromptStructureFeatures) -> PromptStructureLabel:
        if self._fitted and self._model is not None:
            label = self._model.predict([_to_vector(features)])[0]
            return PromptStructureLabel(label)
        return _heuristic_predict(features)

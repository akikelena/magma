"""ML half of the Version Coherence Agent's conflict identification: an
XGBoost gradient-boosted classifier over tabular maintenance-history
features.

Chosen over a graph neural network because the relevant signals form a
small set of tabular features, and gradient-boosted trees match or exceed
deep/graph models on tabular data at this scale. `xgboost` is imported
lazily so this module -- and the `HeuristicConflictScorer` fallback below
-- can be used without the dependency installed, e.g. before a model has
been trained on real episode history.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class ConflictFeatures:
    """One row of the conflict classifier's tabular feature set."""

    change_type_ordinal: int  # ChangeType, ascending severity
    affected_node_count: int
    graph_distance: int  # shortest-path distance between affected components
    historical_co_modification_frequency: float  # in [0, 1]


class ConflictScorer(abc.ABC):
    """Returns a calibrated conflict probability in [0, 1]."""

    @abc.abstractmethod
    def score(self, features: ConflictFeatures) -> float: ...


class HeuristicConflictScorer(ConflictScorer):
    """Deterministic, dependency-free stand-in used before a model has been
    trained on real maintenance-episode history, and in unit tests. Combines
    the four features with hand-set weights; replace with
    `XGBoostConflictScorer` once labeled episodes are available."""

    def score(self, features: ConflictFeatures) -> float:
        severity_component = min(features.change_type_ordinal / 5.0, 1.0) * 0.35
        breadth_component = min(features.affected_node_count / 10.0, 1.0) * 0.25
        proximity_component = (1.0 / (1.0 + features.graph_distance)) * 0.15
        history_component = features.historical_co_modification_frequency * 0.25
        return round(min(severity_component + breadth_component + proximity_component + history_component, 1.0), 4)


class XGBoostConflictScorer(ConflictScorer):
    """Production scorer: a trained XGBoost booster loaded from
    `config/version_coherence_agent.xgboost_model_path`."""

    def __init__(self, model_path: str) -> None:
        import xgboost  # noqa: F401

        self._booster = xgboost.Booster()
        self._booster.load_model(model_path)

    def score(self, features: ConflictFeatures) -> float:
        import xgboost

        dmatrix = xgboost.DMatrix(
            [[features.change_type_ordinal, features.affected_node_count,
              features.graph_distance, features.historical_co_modification_frequency]]
        )
        return float(self._booster.predict(dmatrix)[0])


def weighted_decision(rule_triggered: bool, ml_probability: float, *, threshold: float = 0.5) -> bool:
    """Combine the rule outcome with the calibrated ML probability.

    Any rule-based hit is a hard conflict (rules encode certainties, e.g. a
    real graph cycle). Otherwise the calibrated probability is compared
    against the configurable decision threshold.
    """
    return rule_triggered or ml_probability >= threshold

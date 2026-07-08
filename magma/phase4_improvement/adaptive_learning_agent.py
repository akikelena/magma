"""Adaptive Learning Agent.

Modality: hybrid, combining supervised learning with feedback-driven
policy learning. Supervised learning predicts effective prompt structures
from change characteristics and fine-tunes generation models on validated
outputs of successful episodes. The policy that governs workflow routing
and escalation thresholds is learned from operational feedback as a
contextual bandit (LinUCB), over a reward combining maintenance success
rate, time to resolution, and human-intervention frequency.

A contextual-bandit formulation is preferred over full reinforcement
learning: each routing/escalation decision is a single step with an
immediately observed, directly attributable reward, and available episode
counts are far below what policy-gradient RL needs for stable convergence.

Dependencies: updates are validated offline on held-out sets, released
through canary rollouts monitored for regressions, and rolled back
automatically when performance degrades beyond a configured threshold.

Failure: a rollback event is logged with diagnostic information for
subsequent analysis.

This module covers both halves of the agent's supervised-learning
responsibility: `predict_prompt_structure` predicts an effective prompt
structure from change characteristics
(`PromptStructurePredictor`), and `record_episode_outcome` /
`dispatch_fine_tuning` collect validated successful episodes into a
fine-tuning dataset (`FineTuningDatasetCollector`) and hand it to a
pluggable backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magma.core.governance.audit_log import AuditLogger
from magma.core.interfaces.agent import Agent, AgentResult, Modality
from magma.phase4_improvement.fine_tuning import FineTuningBackend, FineTuningDatasetCollector
from magma.phase4_improvement.prompt_structure import (
    PromptStructureExample,
    PromptStructureFeatures,
    PromptStructureLabel,
    PromptStructurePredictor,
)


@dataclass
class BanditReward:
    success: bool
    time_to_resolution_seconds: float
    human_intervention_count: int


def compute_reward(
    outcome: BanditReward, *, time_scale_seconds: float = 3600.0, intervention_penalty: float = 0.2
) -> float:
    """Combine success rate, resolution time, and intervention frequency
    into a single scalar reward in roughly [-1, 1]."""
    success_term = 1.0 if outcome.success else -1.0
    time_penalty = min(outcome.time_to_resolution_seconds / time_scale_seconds, 1.0) * 0.3
    intervention_term = outcome.human_intervention_count * intervention_penalty
    return success_term - time_penalty - intervention_term


class LinUCBBandit:
    """A minimal LinUCB contextual bandit over a fixed set of discrete arms
    (e.g. execution modes, or "rework vs escalate immediately")."""

    def __init__(self, arm_names: list[str], context_dim: int, alpha: float = 1.0) -> None:
        self.arm_names = arm_names
        self.alpha = alpha
        self.A = {name: np.identity(context_dim) for name in arm_names}
        self.b = {name: np.zeros(context_dim) for name in arm_names}

    def select_arm(self, context: list[float]) -> str:
        x = np.array(context, dtype=float)
        best_arm, best_score = None, float("-inf")
        for name in self.arm_names:
            A_inv = np.linalg.inv(self.A[name])
            theta = A_inv @ self.b[name]
            score = float(theta @ x + self.alpha * np.sqrt(x @ A_inv @ x))
            if score > best_score:
                best_arm, best_score = name, score
        return best_arm

    def update(self, arm: str, context: list[float], reward: float) -> None:
        x = np.array(context, dtype=float)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


@dataclass
class AdaptiveLearningRequest:
    context: list[float]
    observed_arm: str | None = None
    observed_reward: BanditReward | None = None


@dataclass
class AdaptiveLearningOutcome:
    recommended_arm: str
    updated: bool
    rolled_back: bool = False
    rollback_reason: str | None = None


class AdaptiveLearningAgent(Agent[AdaptiveLearningRequest, AdaptiveLearningOutcome]):
    name = "Adaptive Learning Agent"
    modality = Modality.HYBRID

    def __init__(
        self,
        arm_names: list[str],
        context_dim: int,
        alpha: float = 1.0,
        regression_threshold: float = -0.5,
        fine_tuning_backend: FineTuningBackend | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        super().__init__(blackboard=None)
        self.bandit = LinUCBBandit(arm_names=arm_names, context_dim=context_dim, alpha=alpha)
        self.regression_threshold = regression_threshold
        self.audit = audit_logger or AuditLogger()
        self._reward_history: list[float] = []
        # Supervised-learning half: prompt-structure prediction and
        # fine-tuning dataset collection.
        self.prompt_structure_predictor = PromptStructurePredictor()
        self._prompt_structure_examples: list[PromptStructureExample] = []
        self.fine_tuning_collector = (
            FineTuningDatasetCollector(backend=fine_tuning_backend)
            if fine_tuning_backend is not None
            else FineTuningDatasetCollector()
        )

    def predict_prompt_structure(self, features: PromptStructureFeatures) -> PromptStructureLabel:
        """Predict which prompt structure is most likely to lead to a
        successful episode for a change with these features."""
        return self.prompt_structure_predictor.predict(features)

    def record_prompt_structure_outcome(
        self, features: PromptStructureFeatures, label: PromptStructureLabel, success: bool
    ) -> None:
        """Feed one episode's outcome back into the prompt-structure
        training set and refit once enough successful examples exist."""
        self._prompt_structure_examples.append(PromptStructureExample(features=features, label=label, success=success))
        self.prompt_structure_predictor.fit(self._prompt_structure_examples)

    def record_episode_outcome(self, *, episode_id: str, template_id: str, prompt: str, completion: str, success: bool) -> None:
        """Collects a validated episode's (prompt, completion) pair as a
        fine-tuning candidate; only successful episodes are retained."""
        self.fine_tuning_collector.record_episode(
            episode_id=episode_id, template_id=template_id, prompt=prompt, completion=completion, success=success
        )

    def dispatch_fine_tuning(self, *, min_batch_size: int = 1) -> str | None:
        """Submit accumulated successful-episode examples for fine-tuning.
        Returns the backend's job handle, or None if nothing was pending."""
        job_id = self.fine_tuning_collector.dispatch(min_batch_size=min_batch_size)
        if job_id is not None:
            self.audit.log(self.name, "dispatch_fine_tuning", episode_id="adaptive-learning", job_id=job_id)
        return job_id

    def _canary_check(self, reward: float) -> tuple[bool, str | None]:
        """Roll back if the just-observed reward regresses far below the
        recent baseline -- a simple canary-monitoring stand-in for the
        offline-validation + staged-rollout process."""
        if len(self._reward_history) < 5:
            return False, None
        baseline = sum(self._reward_history[-5:]) / 5
        if reward < baseline + self.regression_threshold:
            return True, f"reward {reward:.2f} regressed below baseline {baseline:.2f} + threshold"
        return False, None

    def process(self, data: AdaptiveLearningRequest) -> AgentResult[AdaptiveLearningOutcome]:
        updated = False
        rolled_back = False
        rollback_reason = None

        if data.observed_arm is not None and data.observed_reward is not None:
            reward = compute_reward(data.observed_reward)
            rolled_back, rollback_reason = self._canary_check(reward)
            if not rolled_back:
                self.bandit.update(data.observed_arm, data.context, reward)
                updated = True
                self._reward_history.append(reward)
            else:
                self.audit.log(
                    self.name, "rollback", episode_id=data.observed_arm,
                    reward=reward, reason=rollback_reason,
                )

        recommended = self.bandit.select_arm(data.context)
        self.audit.log(
            self.name, "recommend_arm", episode_id="adaptive-learning",
            recommended_arm=recommended, updated=updated,
        )
        outcome = AdaptiveLearningOutcome(
            recommended_arm=recommended, updated=updated, rolled_back=rolled_back, rollback_reason=rollback_reason
        )
        return AgentResult(agent_name=self.name, modality=self.modality, output=outcome, success=True)

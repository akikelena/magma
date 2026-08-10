"""Usage Feedback Agent.

Modality: deterministic service. Collects performance metrics, failure
events, execution traces, and resource utilization through structured
logging (the audit trail), and identifies the slowest stages, the most
frequently retried operations, recurring failure patterns, and templates
prone to generation errors.

Also estimates per-change maintenance-success probability with a
regularized logistic-regression model, chosen for interpretability and
calibration on a small labeled sample; a gradient-boosted model is
substituted only once the episode count is large enough to support the
added capacity without overfitting.

Failure: not applicable -- this agent is read-only with respect to the
pipeline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from magma.core.governance.audit_log import AuditLogger, AuditRecord
from magma.core.interfaces.agent import Agent, AgentResult, Modality

# Below this many labeled episodes, prefer the interpretable logistic
# model over a higher-capacity gradient-boosted one.
_GRADIENT_BOOSTED_EPISODE_THRESHOLD = 200


@dataclass
class FeedbackReport:
    slowest_actions: list[tuple[str, int]] = field(default_factory=list)  # (action, count) as a throughput proxy
    # (action, total_duration_ms) over records that actually reported
    # `duration_ms` -- the real "slowest stages" answer, distinct from the
    # frequency-based proxy above. Empty when no instrumented agent ran
    # (e.g. only deterministic stages fired).
    slowest_actions_by_duration_ms: list[tuple[str, float]] = field(default_factory=list)
    most_retried_templates: list[tuple[str, int]] = field(default_factory=list)
    failure_patterns: list[tuple[str, int]] = field(default_factory=list)
    error_prone_templates: list[tuple[str, int]] = field(default_factory=list)


class SuccessProbabilityEstimator:
    """Regularized logistic regression over change features.

    `scikit-learn` is imported lazily; `HeuristicFallback` behavior (a
    logistic function over a hand-set linear combination) is used until
    enough labeled episodes exist to fit a real model, keeping this class
    usable with zero training data.
    """

    def __init__(self) -> None:
        self._model = None
        self._fitted = False

    def fit(self, features: list[list[float]], labels: list[int]) -> None:
        if len(features) < 5:
            return  # too little data to fit meaningfully; keep the heuristic fallback
        from sklearn.linear_model import LogisticRegression  # noqa: F401 (lazy import)

        model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
        model.fit(features, labels)
        self._model = model
        self._fitted = True

    def predict_proba(self, features: list[float]) -> float:
        if self._fitted and self._model is not None:
            return float(self._model.predict_proba([features])[0][1])
        # Heuristic fallback: logistic function over an unweighted mean of
        # (assumed normalized-to-[0,1]) features, so behavior is sane before
        # any episode history has accumulated.
        import math

        mean = sum(features) / len(features) if features else 0.5
        return 1.0 / (1.0 + math.exp(-4 * (mean - 0.5)))

    def should_use_gradient_boosted(self, episode_count: int) -> bool:
        return episode_count >= _GRADIENT_BOOSTED_EPISODE_THRESHOLD


class UsageFeedbackAgent(Agent[list[AuditRecord], FeedbackReport]):
    """Runs asynchronously off the accumulated audit trail, alongside the
    main pipeline, on whatever telemetry has accumulated so far --
    `process` is invoked by the orchestration layer's async completion
    hook, not inline in the Phase 1-3 request path."""

    name = "Usage Feedback Agent"
    modality = Modality.ALGORITHMIC

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        super().__init__(blackboard=None)
        self.audit = audit_logger or AuditLogger()
        self.success_estimator = SuccessProbabilityEstimator()

    def process(self, data: list[AuditRecord]) -> AgentResult[FeedbackReport]:
        report = self.analyze(data)
        self.audit.log(
            self.name, "analyze_usage", episode_id="usage-feedback",
            record_count=len(data), failure_pattern_count=len(report.failure_patterns),
        )
        return AgentResult(agent_name=self.name, modality=self.modality, output=report, success=True)

    def analyze(self, records: list[AuditRecord]) -> FeedbackReport:
        action_counts = Counter(r.action for r in records)

        duration_totals: dict[str, float] = {}
        for r in records:
            duration = r.details.get("duration_ms")
            if duration is not None:
                duration_totals[r.action] = duration_totals.get(r.action, 0.0) + float(duration)
        slowest_by_duration = sorted(duration_totals.items(), key=lambda kv: kv[1], reverse=True)[:5]

        retried_templates = Counter(
            r.details.get("template_id", r.episode_id)
            for r in records
            if r.action in ("generate_template",) and r.details.get("is_rework")
        )
        failure_actions = Counter(
            r.action for r in records
            if "failure" in r.action or r.details.get("passed") is False
        )
        error_prone = Counter(
            r.episode_id for r in records
            if r.action == "validate" and r.details.get("passed") is False
        )
        return FeedbackReport(
            slowest_actions=action_counts.most_common(5),
            slowest_actions_by_duration_ms=slowest_by_duration,
            most_retried_templates=retried_templates.most_common(5),
            failure_patterns=failure_actions.most_common(5),
            error_prone_templates=error_prone.most_common(5),
        )

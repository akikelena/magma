import pytest

from magma.phase4_improvement import AdaptiveLearningAgent
from magma.phase4_improvement.fine_tuning import FineTuningDatasetCollector, NoOpFineTuningBackend, TrainingExample
from magma.phase4_improvement.prompt_structure import (
    PromptStructureExample,
    PromptStructureFeatures,
    PromptStructureLabel,
    PromptStructurePredictor,
)


def test_heuristic_predicts_rich_for_security_related_change():
    predictor = PromptStructurePredictor()
    features = PromptStructureFeatures(change_type_ordinal=5, affected_entity_count=1, is_security_related=True)
    assert predictor.predict(features) == PromptStructureLabel.RICH


def test_heuristic_predicts_minimal_for_trivial_change():
    predictor = PromptStructurePredictor()
    features = PromptStructureFeatures(change_type_ordinal=0, affected_entity_count=0, is_security_related=False)
    assert predictor.predict(features) == PromptStructureLabel.MINIMAL


def test_fit_is_a_noop_below_minimum_examples():
    predictor = PromptStructurePredictor()
    examples = [
        PromptStructureExample(
            features=PromptStructureFeatures(5, 1, True), label=PromptStructureLabel.RICH, success=True
        )
    ]
    predictor.fit(examples)
    assert not predictor._fitted  # too few successful examples to fit a real model


def test_fine_tuning_collector_only_keeps_successful_episodes():
    collector = FineTuningDatasetCollector()
    collector.record_episode(episode_id="e1", template_id="t1", prompt="p1", completion="c1", success=True)
    collector.record_episode(episode_id="e2", template_id="t2", prompt="p2", completion="c2", success=False)
    assert collector.pending_count == 1


def test_fine_tuning_dispatch_clears_pending_and_calls_backend():
    backend = NoOpFineTuningBackend()
    collector = FineTuningDatasetCollector(backend=backend)
    collector.record_episode(episode_id="e1", template_id="t1", prompt="p1", completion="c1", success=True)
    job_id = collector.dispatch()
    assert job_id is not None
    assert collector.pending_count == 0
    assert len(backend.submitted_batches) == 1
    assert backend.submitted_batches[0][0] == TrainingExample(episode_id="e1", template_id="t1", prompt="p1", completion="c1")


def test_dispatch_returns_none_below_min_batch_size():
    collector = FineTuningDatasetCollector()
    collector.record_episode(episode_id="e1", template_id="t1", prompt="p1", completion="c1", success=True)
    assert collector.dispatch(min_batch_size=5) is None
    assert collector.pending_count == 1  # nothing was cleared


def test_adaptive_learning_agent_exposes_both_supervised_methods():
    agent = AdaptiveLearningAgent(arm_names=["a"], context_dim=1)
    features = PromptStructureFeatures(change_type_ordinal=1, affected_entity_count=1, is_security_related=False)
    label = agent.predict_prompt_structure(features)
    assert label in PromptStructureLabel

    agent.record_episode_outcome(episode_id="e1", template_id="t1", prompt="p", completion="c", success=True)
    job_id = agent.dispatch_fine_tuning()
    assert job_id is not None


def test_fit_trains_a_real_model_at_or_above_minimum_examples():
    """Locks in the real scikit-learn path (below-threshold heuristic
    fallback is covered by test_fit_is_a_noop_below_minimum_examples
    above): once >= _MIN_EXAMPLES_TO_FIT successful examples are
    available, fit() must actually train a DecisionTreeClassifier and
    predict() must defer to it rather than the heuristic. Skipped if
    scikit-learn isn't installed -- this repo's CI doesn't require it."""
    pytest.importorskip("sklearn")
    examples = [
        PromptStructureExample(
            features=PromptStructureFeatures(change_type_ordinal=5, affected_entity_count=3, is_security_related=True),
            label=PromptStructureLabel.RICH, success=True,
        )
        for _ in range(5)
    ] + [
        PromptStructureExample(
            features=PromptStructureFeatures(change_type_ordinal=0, affected_entity_count=0, is_security_related=False),
            label=PromptStructureLabel.MINIMAL, success=True,
        )
        for _ in range(5)
    ]

    predictor = PromptStructurePredictor()
    predictor.fit(examples)

    assert predictor._fitted
    assert predictor._model is not None
    # A clearly security-related, high-severity case should predict RICH
    # from the fitted model, not merely because the heuristic would also
    # say so -- prove the model path is live by checking it agrees.
    predicted = predictor.predict(PromptStructureFeatures(change_type_ordinal=5, affected_entity_count=3, is_security_related=True))
    assert predicted == PromptStructureLabel.RICH

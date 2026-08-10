from magma.phase4_improvement import AdaptiveLearningAgent
from magma.phase4_improvement.adaptive_learning_agent import AdaptiveLearningRequest, BanditReward, compute_reward


def test_reward_favors_success_and_speed():
    good = compute_reward(BanditReward(success=True, time_to_resolution_seconds=60, human_intervention_count=0))
    bad = compute_reward(BanditReward(success=False, time_to_resolution_seconds=7200, human_intervention_count=2))
    assert good > bad


def test_recommendation_without_update_does_not_mutate_state():
    agent = AdaptiveLearningAgent(arm_names=["a", "b"], context_dim=2)
    result = agent.process(AdaptiveLearningRequest(context=[1.0, 1.0]))
    assert not result.output.updated
    assert result.output.recommended_arm in ("a", "b")


def test_update_learns_toward_rewarded_arm():
    agent = AdaptiveLearningAgent(arm_names=["a", "b"], context_dim=2, alpha=0.0)
    context = [1.0, 0.0]
    for _ in range(10):
        agent.process(AdaptiveLearningRequest(
            context=context, observed_arm="a",
            observed_reward=BanditReward(success=True, time_to_resolution_seconds=10, human_intervention_count=0),
        ))
    result = agent.process(AdaptiveLearningRequest(context=context))
    assert result.output.recommended_arm == "a"


def test_reward_regression_triggers_rollback():
    agent = AdaptiveLearningAgent(arm_names=["a"], context_dim=1, regression_threshold=-0.1)
    good_reward = BanditReward(success=True, time_to_resolution_seconds=10, human_intervention_count=0)
    for _ in range(5):
        agent.process(AdaptiveLearningRequest(context=[1.0], observed_arm="a", observed_reward=good_reward))
    bad_reward = BanditReward(success=False, time_to_resolution_seconds=7200, human_intervention_count=5)
    result = agent.process(AdaptiveLearningRequest(context=[1.0], observed_arm="a", observed_reward=bad_reward))
    assert result.output.rolled_back
    assert not result.output.updated

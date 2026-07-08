from magma.phase4_improvement import SyntheticTestingAgent
from magma.phase4_improvement.synthetic_testing_agent import AdversarialScenario


def test_all_scenarios_survive():
    agent = SyntheticTestingAgent()
    result = agent.process([AdversarialScenario("s1", lambda: True), AdversarialScenario("s2", lambda: True)])
    assert result.output.survival_rate == 1.0
    assert result.output.failure_patterns == []


def test_failing_and_raising_scenarios_are_captured_not_raised():
    def boom():
        raise ValueError("boom")

    agent = SyntheticTestingAgent()
    result = agent.process([AdversarialScenario("fails", lambda: False), AdversarialScenario("raises", boom)])
    assert result.success  # the agent itself must not raise
    assert result.output.survival_rate == 0.0
    assert set(result.output.failure_patterns) == {"fails", "raises"}


def test_default_scenarios_survive_against_real_agents():
    """default_scenarios() builds its battery against real
    ChangeDetectionAgent/VersionCoherenceAgent instances but was never
    actually invoked anywhere in the codebase or test suite -- this
    locks in that the real integration (not just SyntheticTestingAgent's
    own isolated toy-lambda tests above) actually works end to end."""
    from magma.knowledge_repository import InMemoryGraphStore
    from magma.llm_providers import MockLLMProvider
    from magma.phase1_documentation_detection.change_detection_agent import ChangeDetectionAgent
    from magma.phase4_improvement.synthetic_testing_agent import default_scenarios

    change_detection_agent = ChangeDetectionAgent(llm_provider=MockLLMProvider(responder=lambda p: "not valid json"))
    graph_store = InMemoryGraphStore()

    scenarios = default_scenarios(change_detection_agent, None, graph_store)
    assert len(scenarios) == 3

    agent = SyntheticTestingAgent()
    result = agent.process(scenarios)

    assert result.success
    assert result.output.survival_rate == 1.0, result.output.failure_patterns


def test_default_scenarios_uses_the_passed_in_coherence_agent_not_a_fresh_one():
    """Locks in the fix: a custom version_coherence_agent instance
    (e.g. wired to a live graph with a non-default conflict scorer) must
    actually be used by the coherence scenario, not silently replaced
    with a fresh vanilla VersionCoherenceAgent(graph_store=...)."""
    from magma.knowledge_repository import InMemoryGraphStore
    from magma.llm_providers import MockLLMProvider
    from magma.phase1_documentation_detection.change_detection_agent import ChangeDetectionAgent
    from magma.phase1_documentation_detection.version_coherence_agent import VersionCoherenceAgent
    from magma.phase4_improvement.synthetic_testing_agent import default_scenarios

    calls = []

    class _SpyCoherenceAgent(VersionCoherenceAgent):
        def process(self, data):
            calls.append(data)
            return super().process(data)

    graph_store = InMemoryGraphStore()
    spy = _SpyCoherenceAgent(graph_store=graph_store)
    change_detection_agent = ChangeDetectionAgent(llm_provider=MockLLMProvider(responder=lambda p: "not valid json"))

    scenarios = default_scenarios(change_detection_agent, spy, graph_store)
    coherence_scenario = next(s for s in scenarios if s.name == "coherence_with_no_changes")
    coherence_scenario.run()

    assert len(calls) == 1, "the passed-in VersionCoherenceAgent instance was never called"

"""Shared pytest fixtures.

`fake_crewai` provides a minimal stand-in for the `crewai` package so
magma.crewai_agents.* can be exercised without the real (heavy)
dependency installed -- this repo's CI never requires crewai/langchain,
matching the lazy-import pattern used for every other optional backend
(neo4j, weaviate, xgboost, sklearn, spacy, prometheus_client). The fake
only needs to reproduce CrewAI's *structural* contract (Agent/Task/Crew/
Process, and Process.sequential running each task's agent's tool in
order) -- the actual business logic under test is MAGMA's own tool
closures, not CrewAI's real LLM-driven scheduling, which this repo does
not own.
"""
from __future__ import annotations

import sys
import types

import pytest


class _FakeProcess:
    sequential = "sequential"


class _FakeAgent:
    def __init__(self, role, goal="", backstory="", tools=None, **settings):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        # Real crewai lets an LLM pick a tool by name from the ones
        # offered; this mirrors that lookup for tests that need to call a
        # specific tool out of a multi-tool agent rather than assuming
        # position 0.
        self.tools_by_name = {getattr(t, "__name__", str(i)): t for i, t in enumerate(self.tools)}
        # Everything CrewAgentSettings.as_kwargs() contributes (verbose,
        # allow_delegation, max_iter, ...) lands here so tests can assert
        # it was actually passed through, without this fake having to
        # mirror crewai's full Agent signature.
        self.settings = settings


class _FakeTask:
    def __init__(self, description, agent, expected_output="", context=None):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.context = context or []


class _FakeCrew:
    def __init__(self, agents, tasks, process, **settings):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.settings = settings
        self.kickoff_outputs: list = []

    def kickoff(self, planner=None):
        """Naive sequential execution matching Process.sequential: call
        each task's agent's sole tool, in task list order, unless
        `planner` is given.

        `planner(agent, task) -> Any` is a test-only stand-in for real
        LLM tool selection -- for an agent with several tools (e.g.
        phase1's Documentation Change Detector), a test can script the
        realistic call sequence a competent LLM would make and prove that
        sequence produces the correct end-to-end result, without a live
        LLM. Agents with exactly one tool don't need it: the default
        (call tools[0]) already exercises them.
        """
        if planner is not None:
            self.kickoff_outputs = [planner(task.agent, task) for task in self.tasks]
        else:
            self.kickoff_outputs = [task.agent.tools[0]() for task in self.tasks]
        return self.kickoff_outputs


@pytest.fixture
def fake_crewai(monkeypatch):
    fake = types.ModuleType("crewai")
    fake.Agent = _FakeAgent
    fake.Task = _FakeTask
    fake.Crew = _FakeCrew
    fake.Process = _FakeProcess
    monkeypatch.setitem(sys.modules, "crewai", fake)
    return fake

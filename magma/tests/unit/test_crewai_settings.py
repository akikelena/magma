"""Unit coverage for magma.crewai_agents.settings, independent of the
crewai_agents integration suite (which exercises these through the crew
builders against fake_crewai)."""
from __future__ import annotations

from magma.core.utils.config import MagmaConfig
from magma.crewai_agents.settings import CrewAgentSettings, CrewSettings, load_agent_settings, load_crew_settings


def test_load_agent_settings_uses_injected_config_over_the_committed_yaml():
    config = MagmaConfig(crewai={"phase2": {"max_iter": 42, "max_rpm": 10}})
    settings = load_agent_settings("phase2", config=config)
    assert settings.max_iter == 42
    assert settings.max_rpm == 10
    assert settings.cache is True  # unspecified fields keep their dataclass default


def test_load_agent_settings_unknown_phase_returns_all_defaults():
    config = MagmaConfig(crewai={})
    settings = load_agent_settings("phase99", config=config)
    assert settings == CrewAgentSettings()


def test_load_crew_settings_uses_injected_config():
    config = MagmaConfig(crewai={"crew": {"memory": True, "planning": True}})
    settings = load_crew_settings(config=config)
    assert settings.memory is True
    assert settings.planning is True
    assert settings.cache is True


def test_crew_agent_settings_is_frozen_and_hashable():
    settings = CrewAgentSettings(max_iter=5)
    assert hash(settings)  # frozen dataclasses are hashable -- must stay safe to use as a dict default


def test_crew_settings_as_kwargs_round_trips_through_from_dict():
    original = CrewSettings(memory=True, max_rpm=30, output_log_file="run.log")
    rebuilt = CrewSettings.from_dict(original.as_kwargs())
    assert rebuilt == original

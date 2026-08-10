"""magma.core.utils.config was orphaned: `load_config()`/`MagmaConfig`
were defined, exported, and had real YAML fixtures at config/agents.yaml
and config/llm_models.yaml, but nothing in the codebase called them until
run_workflow.py's CLI defaults were wired to it. Writing this test
surfaced a real bug in the process: `DEFAULT_CONFIG_DIR` resolved
`Path(__file__).parents[3]` (the repo root, one level too high) instead
of `parents[2]` (`magma/config`), so `load_config()` with no args always
silently returned empty dicts -- `_load_yaml` treats a missing file as
"no config" rather than raising, which is exactly why the bug went
unnoticed instead of crashing the first time anyone called it."""
from __future__ import annotations

from pathlib import Path

from magma.core.utils.config import DEFAULT_CONFIG_DIR, MagmaConfig, load_config


def test_default_config_dir_points_at_the_real_committed_config_directory():
    assert DEFAULT_CONFIG_DIR.name == "config"
    assert (DEFAULT_CONFIG_DIR / "agents.yaml").is_file()
    assert (DEFAULT_CONFIG_DIR / "llm_models.yaml").is_file()


def test_load_config_with_no_args_reads_the_real_files_not_empty_dicts():
    config = load_config()
    assert config.agents, "load_config() returned an empty agents dict -- DEFAULT_CONFIG_DIR is wrong again"
    assert config.llm_models, "load_config() returned an empty llm_models dict -- DEFAULT_CONFIG_DIR is wrong again"
    assert "change_detection_agent" in config.agents


def test_load_config_accepts_an_explicit_config_dir(tmp_path: Path):
    (tmp_path / "agents.yaml").write_text("some_agent:\n  model: some_alias\n  threshold: 0.1\n")
    (tmp_path / "llm_models.yaml").write_text("some_alias:\n  provider: mock\n  model: mock-1\n")

    config = load_config(tmp_path)
    assert config.agent("some_agent") == {"model": "some_alias", "threshold": 0.1}
    assert config.model_for("some_agent") == {"provider": "mock", "model": "mock-1"}


def test_load_config_missing_directory_yields_empty_config_not_a_crash(tmp_path: Path):
    """Documents the intentional fallback behavior: a missing config
    directory (e.g. a stripped-down deployment) degrades to an empty
    MagmaConfig rather than raising, so callers must handle `agent()`/
    `model_for()` returning `{}`."""
    config = load_config(tmp_path / "does_not_exist")
    assert config == MagmaConfig(agents={}, llm_models={})
    assert config.agent("anything") == {}
    assert config.model_for("anything") == {}


def test_model_for_falls_back_to_default_alias_when_agent_has_no_model():
    config = MagmaConfig(
        agents={"deterministic_agent": {"model": None}},
        llm_models={"default": {"provider": "anthropic", "model": "claude-sonnet-4-5"}},
    )
    assert config.model_for("deterministic_agent") == {"provider": "anthropic", "model": "claude-sonnet-4-5"}


def test_model_for_unknown_agent_falls_back_to_default_alias():
    config = MagmaConfig(
        agents={},
        llm_models={"default": {"provider": "anthropic", "model": "claude-sonnet-4-5"}},
    )
    assert config.model_for("never_configured_agent") == {"provider": "anthropic", "model": "claude-sonnet-4-5"}

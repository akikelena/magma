"""YAML-based configuration loading (config/agents.yaml, config/llm_models.yaml,
config/crewai.yaml, config/vendor_sources.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass
class MagmaConfig:
    agents: dict[str, Any] = field(default_factory=dict)
    llm_models: dict[str, Any] = field(default_factory=dict)
    crewai: dict[str, Any] = field(default_factory=dict)
    vendor_sources: dict[str, Any] = field(default_factory=dict)

    def agent(self, name: str) -> dict[str, Any]:
        return self.agents.get(name, {})

    def model_for(self, agent_name: str) -> dict[str, Any]:
        alias = self.agent(agent_name).get("model", "default")
        return self.llm_models.get(alias, self.llm_models.get("default", {}))

    def crewai_agent(self, phase: str) -> dict[str, Any]:
        return self.crewai.get(phase, {})

    def crewai_crew(self) -> dict[str, Any]:
        return self.crewai.get("crew", {})

    def vendor_source(self, vendor: str) -> dict[str, Any]:
        return self.vendor_sources.get(vendor, {})


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str | Path | None = None) -> MagmaConfig:
    directory = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    return MagmaConfig(
        agents=_load_yaml(directory / "agents.yaml"),
        llm_models=_load_yaml(directory / "llm_models.yaml"),
        crewai=_load_yaml(directory / "crewai.yaml"),
        vendor_sources=_load_yaml(directory / "vendor_sources.yaml"),
    )

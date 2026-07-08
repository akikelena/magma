"""Configurable CrewAI Agent/Crew attributes.

Every build_phaseN_crew()/build_magma_crew() previously hardcoded its
crewai.Agent/Crew calls to bare defaults (role/goal/backstory/tools only).
CrewAgentSettings and CrewSettings expose CrewAI's real tunables --
iteration/time/rate limits, retry policy, caching, delegation -- as MAGMA
config, sourced from config/crewai.yaml, so operators can tune scheduling
per phase or for the whole crew without editing crew-builder code.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from magma.core.utils.config import MagmaConfig, load_config


@dataclass(frozen=True)
class CrewAgentSettings:
    verbose: bool = False
    allow_delegation: bool = False
    allow_code_execution: bool = False
    cache: bool = True
    respect_context_window: bool = True
    max_iter: int = 20
    max_retry_limit: int = 2
    max_rpm: int | None = None
    max_execution_time: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrewAgentSettings":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def as_kwargs(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


@dataclass(frozen=True)
class CrewSettings:
    verbose: bool = False
    memory: bool = False
    cache: bool = True
    planning: bool = False
    full_output: bool = False
    max_rpm: int | None = None
    output_log_file: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrewSettings":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def as_kwargs(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


def load_agent_settings(phase: str, config: MagmaConfig | None = None) -> CrewAgentSettings:
    config = config or load_config()
    return CrewAgentSettings.from_dict(config.crewai_agent(phase))


def load_crew_settings(config: MagmaConfig | None = None) -> CrewSettings:
    config = config or load_config()
    return CrewSettings.from_dict(config.crewai_crew())

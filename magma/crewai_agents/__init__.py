from magma.crewai_agents.magma_crew import build_magma_crew
from magma.crewai_agents.phase1_crew import build_phase1_crew
from magma.crewai_agents.phase2_crew import build_phase2_crew
from magma.crewai_agents.phase3_crew import build_phase3_crew
from magma.crewai_agents.phase4_crew import build_phase4_crew
from magma.crewai_agents.settings import CrewAgentSettings, CrewSettings, load_agent_settings, load_crew_settings

__all__ = [
    "build_phase1_crew",
    "build_phase2_crew",
    "build_phase3_crew",
    "build_phase4_crew",
    "build_magma_crew",
    "CrewAgentSettings",
    "CrewSettings",
    "load_agent_settings",
    "load_crew_settings",
]

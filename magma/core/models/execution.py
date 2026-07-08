"""Execution-planning models shared by Phase 2 (Task Orchestration, Prompt
Generation, Code Generation agents)."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class ExecutionMode(str, enum.Enum):
    LIGHTWEIGHT = "lightweight"      # localized single-template change
    COMPREHENSIVE = "comprehensive"  # coordinated updates across templates
    PARALLEL = "parallel"            # independent change sets, concurrent


class MaintenanceCategory(str, enum.Enum):
    ADDITIVE = "additive"
    CORRECTIVE = "corrective"
    TRANSFORMATIVE = "transformative"


class GenerationMode(str, enum.Enum):
    TARGETED_MODIFICATION = "targeted_modification"  # <30% of template lines affected
    COMPREHENSIVE_SYNTHESIS = "comprehensive_synthesis"


@dataclass
class TemplateTask:
    template_id: str
    maintenance_category: MaintenanceCategory
    generation_mode: GenerationMode
    instructions: str = ""
    # Populated by the Task Orchestration Agent's affected-artifact
    # analysis from the template's graph-node properties.
    generation_rules: list[str] = field(default_factory=list)
    output_formats: list[str] = field(default_factory=list)
    architectural_constraints: str = ""


@dataclass
class ExecutionPlan:
    changeset_id: str
    execution_mode: ExecutionMode
    ordered_tasks: list[TemplateTask] = field(default_factory=list)


@dataclass
class GeneratedArtifact:
    template_id: str
    content: str
    documentation_comments: str = ""
    test_cases: str = ""
    generation_mode: GenerationMode = GenerationMode.TARGETED_MODIFICATION

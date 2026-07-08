"""Stage 3 of the Documentation Processing Agent: fragment labeling.

Assigns each parsed section one of the four `FragmentLabel`s using the
entities/relations extracted in stage 2, combined with lexical cues. This
stays a deterministic rule layer (no LLM) per the agent's ALGORITHMIC
modality; the six-way *change* classification is a separate, later
responsibility of the (hybrid) Change Detection Agent.
"""
from __future__ import annotations

import re

from magma.core.models.documentation import FragmentLabel
from magma.phase1_documentation_detection.nlp_pipeline import NLPAnnotation

_DEPRECATION_RE = re.compile(r"\bdeprecat\w+|\bwill be removed\b|\bno longer supported\b", re.IGNORECASE)
_MIGRATION_RE = re.compile(r"\bmigrat\w+|\bupgrad\w+ from\b|\breplace\w* with\b", re.IGNORECASE)
_ERROR_HANDLING_RE = re.compile(r"\bthrows?\b|\braises?\b|\berror\b|\bexception\b|\bfails? if\b", re.IGNORECASE)
_API_CONSTRAINT_RE = re.compile(r"\brequires?\b|\bmust\b|\bmandatory\b|\bonly accepts\b|\bvalid values\b", re.IGNORECASE)


def label_fragment(content: str, annotation: NLPAnnotation) -> FragmentLabel:
    """Rule-based label assignment, checked in order of specificity."""
    if _DEPRECATION_RE.search(content):
        return FragmentLabel.DEPRECATION_NOTICE
    if _MIGRATION_RE.search(content):
        return FragmentLabel.MIGRATION_INSTRUCTION
    if _ERROR_HANDLING_RE.search(content):
        return FragmentLabel.ERROR_HANDLING_REQUIREMENT
    if _API_CONSTRAINT_RE.search(content) or any(e.label == "PARAMETER_TYPE" for e in annotation.entities):
        return FragmentLabel.API_CONSTRAINT
    return FragmentLabel.UNCLASSIFIED

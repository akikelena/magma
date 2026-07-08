"""Chain-of-thought prompt construction and response parsing for the LLM
stage of the Change Detection Agent.

The model is asked to (1) identify affected entities, (2) assess behavioral
impact, (3) compare against API contract semantics, and only then (4)
assign one of six labels. Boundary rule: a type change is MAJOR if the
prior representation remains valid, BREAKING if the prior type is
deprecated/removed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from magma.core.models.change import ChangeType

_SYSTEM_PROMPT = """You are the semantic-classification stage of a documentation change \
detector for auto-generated API client templates. You are given a unified diff between \
two versions of a documentation fragment. Reason step by step:
1. Identify the affected entities (API names, parameters, types, versions).
2. Assess the behavioral impact on code that currently calls this API.
3. Compare the new specification against the old one to determine whether prior \
   generated code remains valid (compatibility is preserved) or would fail (compatibility \
   is broken).
4. Only then assign exactly one category from: EDITORIAL, PATCH, MINOR, MAJOR, \
   DEPRECATION, BREAKING.

Boundary rule: a change is MAJOR if the prior representation still works after the change; \
it is BREAKING if the prior representation is deprecated or removed by the change.

Respond with your reasoning, then a final line containing ONLY a JSON object of the form:
{"change_type": "<ONE_OF_THE_SIX_LABELS>", "confidence": <0..1>, "intent_shift": <0..1>, \
"compatibility_preserved": <true|false>, "affected_entities": ["..."], "rationale": "..."}
"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_prompt(fragment_id: str, edit_script: list[str]) -> str:
    diff_text = "".join(edit_script)
    return (
        f"Fragment: {fragment_id}\n\n"
        f"Unified diff:\n```\n{diff_text}\n```\n\n"
        "Walk through steps 1-3, then output the final JSON object on its own line."
    )


@dataclass
class ParsedClassification:
    change_type: ChangeType
    confidence: float
    intent_shift: float
    compatibility_preserved: bool
    affected_entities: list[str]
    rationale: str


class ClassificationParseError(ValueError):
    pass


def parse_classification(llm_text: str) -> ParsedClassification:
    match = _JSON_OBJECT_RE.search(llm_text)
    if not match:
        raise ClassificationParseError(f"no JSON object found in LLM output: {llm_text!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ClassificationParseError(f"malformed JSON from LLM: {exc}") from exc

    try:
        change_type = ChangeType[str(payload["change_type"]).upper()]
    except KeyError as exc:
        raise ClassificationParseError(f"unknown change_type: {payload.get('change_type')!r}") from exc

    return ParsedClassification(
        change_type=change_type,
        confidence=float(payload.get("confidence", 0.0)),
        intent_shift=float(payload.get("intent_shift", 0.0)),
        compatibility_preserved=bool(payload.get("compatibility_preserved", True)),
        affected_entities=list(payload.get("affected_entities", [])),
        rationale=str(payload.get("rationale", "")),
    )


SYSTEM_PROMPT = _SYSTEM_PROMPT

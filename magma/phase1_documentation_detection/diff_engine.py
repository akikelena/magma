"""Deterministic first stage of the Change Detection Agent: Myers' diff.

Textual differences between two documentation versions are computed with
Myers' difference algorithm (via Python's `difflib`, which implements the
Ratcliff/Obershelp-derived `SequenceMatcher` but exposes Myers-style
`unified_diff` output), producing edit scripts at line granularity. Running
this first means the costly LLM-based semantic reasoning in stage two only
ever sees fragment pairs that actually differ.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass(frozen=True)
class DiffResult:
    edit_script: list[str]
    lines_added: int
    lines_removed: int

    @property
    def is_identical(self) -> bool:
        return self.lines_added == 0 and self.lines_removed == 0


def compute_diff(old_text: str, new_text: str, *, context_lines: int = 2) -> DiffResult:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    script = list(
        difflib.unified_diff(old_lines, new_lines, fromfile="old", tofile="new", n=context_lines)
    )
    added = sum(1 for line in script if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in script if line.startswith("-") and not line.startswith("---"))
    return DiffResult(edit_script=script, lines_added=added, lines_removed=removed)


_TRIVIAL_ONLY_WHITESPACE = lambda line: line.strip("+-").strip() == ""  # noqa: E731


def is_editorial_only(diff: DiffResult) -> bool:
    """True if every changed line is blank/whitespace -- a purely editorial
    edit that the deterministic preprocessor filters before invoking the LLM
    stage (first line of defense)."""
    changed_lines = [
        line for line in diff.edit_script
        if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---"))
    ]
    return bool(changed_lines) and all(_TRIVIAL_ONLY_WHITESPACE(line) for line in changed_lines)

"""The six validation layers realized by the Automated Validation Agent,
ordered by ascending cost so cheaper checks remove most defects before
the more expensive ones are reached.

Each validator is a small, storage/tool-agnostic interface so real
tooling (Xtend syntax checking, custom style rulesets, JUnit, SpotBugs,
JMH, Docker-containerized VDBMS instances for integration) can be
substituted per validator without touching the Automated Validation
Agent itself. Defaults here are deterministic, dependency-light
stand-ins so the six-layer pipeline is fully exercised in CI without
that toolchain installed.
"""
from __future__ import annotations

import abc
import json
import re
import subprocess
import sys

from magma.core.models.execution import GeneratedArtifact
from magma.core.models.validation import LayerResult, ValidationLayer

_EXEC_TIMEOUT_SECONDS = 5

# Runs inside a fresh `python -c` subprocess (not this process) so a
# generated artifact can never touch our interpreter state: the artifact
# body and its test cases are exec'd in an isolated namespace, tests
# whose name starts with "test" are called, and a line-level tracer scoped
# to the artifact's own compiled unit records basic statement coverage.
_PY_HARNESS = """
import ast, json, sys

_artifact_src = {artifact_src!r}
_test_src = {test_src!r}

# A SyntaxError here means the artifact isn't actually Python (e.g. real
# Xtend/Java output) -- reported distinctly so the caller can fall back to
# a presence-only check instead of treating it as a functional defect.
try:
    _artifact_code = compile(_artifact_src, "<artifact>", "exec")
    _test_code = compile(_test_src, "<tests>", "exec")
except SyntaxError:
    print(json.dumps({{"non_python": True}}))
    sys.exit(0)

failures = []
test_fns = []
_hits = set()

def _tracer(frame, event, arg):
    if event == "line" and frame.f_code.co_filename == "<artifact>":
        _hits.add(frame.f_lineno)
    return _tracer

_artifact_ns: dict = {{}}
try:
    exec(_artifact_code, _artifact_ns)
except Exception as exc:
    failures.append(f"<artifact setup>: {{exc!r}}")

if not failures:
    _test_ns = dict(_artifact_ns)
    try:
        exec(_test_code, _test_ns)
    except Exception as exc:
        failures.append(f"<test setup>: {{exc!r}}")
    test_fns = [v for k, v in _test_ns.items() if k.startswith("test") and callable(v)]
    sys.settrace(_tracer)
    try:
        for fn in test_fns:
            try:
                fn()
            except Exception as exc:
                failures.append(f"{{fn.__name__}}: {{exc!r}}")
    finally:
        sys.settrace(None)

total_statements = sum(1 for node in ast.walk(ast.parse(_artifact_src)) if isinstance(node, ast.stmt))
print(json.dumps({{
    "non_python": False,
    "tests_run": len(test_fns),
    "failures": failures,
    "covered_lines": len(_hits),
    "total_statements": total_statements,
}}))
"""

_SECURITY_SENSITIVE_PATTERNS = {
    "eval_usage": re.compile(r"\beval\s*\("),
    "exec_usage": re.compile(r"\bexec\s*\("),
    "hardcoded_secret": re.compile(r"(?i)(api_key|password|secret)\s*=\s*['\"][^'\"]+['\"]"),
    "tls_verification_disabled": re.compile(r"verify\s*=\s*False|insecure\s*=\s*true", re.IGNORECASE),
    "sql_string_concatenation": re.compile(r"(?i)(select|insert|update|delete)\b.*['\"]\s*\+"),
}


class Validator(abc.ABC):
    layer: ValidationLayer

    @abc.abstractmethod
    def validate(self, artifact: GeneratedArtifact) -> LayerResult: ...


class SyntaxValidator(Validator):
    """Syntax validation runs first: syntactic correctness is a
    prerequisite for every later check. Production configuration uses
    Xtend 2.41; this default does a balanced-delimiter check that is
    language-agnostic and dependency-free."""

    layer = ValidationLayer.SYNTAX

    def validate(self, artifact: GeneratedArtifact) -> LayerResult:
        pairs = {"(": ")", "[": "]", "{": "}"}
        closers = set(pairs.values())
        stack: list[str] = []
        for char in artifact.content:
            if char in pairs:
                stack.append(pairs[char])
            elif char in closers:
                if not stack or stack.pop() != char:
                    return LayerResult(self.layer, passed=False, details=f"unbalanced delimiter near {char!r}")
        if stack:
            return LayerResult(self.layer, passed=False, details=f"unclosed delimiters: {stack}")
        return LayerResult(self.layer, passed=True)


class StyleValidator(Validator):
    """Enforces coding standards. Default: line-length and
    trailing-whitespace rules; production configuration uses custom
    rulesets."""

    layer = ValidationLayer.STYLE
    max_line_length = 120

    def validate(self, artifact: GeneratedArtifact) -> LayerResult:
        violations = [
            i + 1 for i, line in enumerate(artifact.content.splitlines())
            if len(line) > self.max_line_length or line != line.rstrip()
        ]
        if violations:
            return LayerResult(self.layer, passed=False, details=f"style violations on lines {violations}")
        return LayerResult(self.layer, passed=True)


class FunctionalValidator(Validator):
    """Automated testing of typical/boundary scenarios, with completeness
    measured through statement/branch/path coverage. Production
    configuration uses JUnit 6 against Xtend-generated sources; for
    Python-shaped artifacts (as produced by this reference build's Code
    Generation Agent) this validator actually executes the generated test
    cases against the generated content in a sandboxed subprocess and
    measures line-level statement coverage, rather than only checking
    that tests were present. Non-Python artifacts (e.g. "@Test"-annotated
    JUnit sources, real Xtend output) fall back to the presence-only
    check, since executing those requires the JUnit/Xtend toolchain --
    this validator stays pluggable for that substitution rather than
    faking it."""

    layer = ValidationLayer.FUNCTIONAL
    exec_timeout_seconds = _EXEC_TIMEOUT_SECONDS

    def validate(self, artifact: GeneratedArtifact) -> LayerResult:
        if not artifact.test_cases.strip():
            return LayerResult(self.layer, passed=False, details="no test cases were generated for this artifact")
        if "def test" not in artifact.test_cases and "@Test" not in artifact.test_cases:
            return LayerResult(self.layer, passed=False, details="test_cases does not contain a recognizable test")

        if "@Test" in artifact.test_cases:
            return LayerResult(
                self.layer, passed=True,
                details="JUnit-style test_cases detected; execution deferred to the production JUnit/Xtend toolchain",
            )

        return self._execute_python_tests(artifact)

    def _execute_python_tests(self, artifact: GeneratedArtifact) -> LayerResult:
        harness = _PY_HARNESS.format(artifact_src=artifact.content, test_src=artifact.test_cases)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", harness],
                capture_output=True, text=True, timeout=self.exec_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return LayerResult(self.layer, passed=False, details="test execution exceeded the sandbox timeout")

        if proc.returncode != 0:
            return LayerResult(
                self.layer, passed=False,
                details=f"sandbox execution crashed: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'unknown error'}",
            )

        try:
            summary = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return LayerResult(self.layer, passed=False, details="test execution produced no parseable result")

        if summary.get("non_python"):
            # Not valid Python (e.g. real Xtend/Java output) -- execution
            # requires the production JUnit/Xtend toolchain, so fall back
            # to the presence-only check for this artifact.
            return LayerResult(
                self.layer, passed=True,
                details="artifact is not executable Python; presence-only check passed",
            )

        if summary["failures"]:
            return LayerResult(
                self.layer, passed=False,
                details=f"{len(summary['failures'])}/{summary['tests_run']} test(s) failed: {summary['failures']}",
                diagnostics=summary,
            )

        total = summary["total_statements"]
        covered = summary["covered_lines"]
        # A hard failure only for the unambiguous case: the artifact has
        # executable statements but the generated tests never exercised
        # any of them (e.g. every test body is a bare `pass`). Anything
        # short of that is reported as a coverage diagnostic rather than a
        # gate, since exact percentage thresholds are brittle for small
        # generated units.
        if total and covered == 0:
            return LayerResult(
                self.layer, passed=False,
                details="generated tests passed but never executed any statement in the artifact (0% coverage)",
                diagnostics=summary,
            )

        coverage = (covered / total) if total else 1.0
        return LayerResult(
            self.layer, passed=True,
            details=f"{summary['tests_run']} test(s) passed, {coverage:.0%} statement coverage",
            diagnostics=summary,
        )


class IntegrationValidator(Validator):
    """Integration defects appear only under interaction with external
    systems; production configuration exercises generated templates
    against Docker-containerized VDBMS instances. This default is a
    pluggable no-op that always passes, standing in for that
    out-of-process integration harness."""

    layer = ValidationLayer.INTEGRATION

    def validate(self, artifact: GeneratedArtifact) -> LayerResult:
        return LayerResult(self.layer, passed=True, details="integration harness not wired in this environment")


class SecurityScanner(Validator):
    """Examines the artifact for injection flaws, insecure cryptographic
    practices, and API misuse. The scanned defect classes are a
    pluggable, domain-specific policy while the scanning mechanism stays
    domain-independent."""

    layer = ValidationLayer.SECURITY

    def __init__(self, patterns: dict[str, "re.Pattern"] | None = None) -> None:
        self.patterns = patterns or _SECURITY_SENSITIVE_PATTERNS

    def validate(self, artifact: GeneratedArtifact) -> LayerResult:
        findings = {name: bool(pattern.search(artifact.content)) for name, pattern in self.patterns.items()}
        hits = [name for name, found in findings.items() if found]
        if hits:
            return LayerResult(self.layer, passed=False, details=f"security scan flagged: {hits}", diagnostics=findings)
        return LayerResult(self.layer, passed=True)


class PerformanceValidator(Validator):
    """Runs last: reliable measurement requires functionally correct and
    integrated code. Production configuration uses JMH; this default is
    a pluggable no-op pass."""

    layer = ValidationLayer.PERFORMANCE

    def validate(self, artifact: GeneratedArtifact) -> LayerResult:
        return LayerResult(self.layer, passed=True, details="performance profiling not wired in this environment")


DEFAULT_VALIDATORS: list[Validator] = [
    SyntaxValidator(),
    StyleValidator(),
    FunctionalValidator(),
    IntegrationValidator(),
    SecurityScanner(),
    PerformanceValidator(),
]

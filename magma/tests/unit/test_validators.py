"""Direct unit tests for the FunctionalValidator's real sandboxed
execution, independent of the six-layer pipeline ordering already
covered by test_automated_validation_agent.py."""
from __future__ import annotations

from magma.core.models.execution import GeneratedArtifact
from magma.core.models.validation import ValidationLayer
from magma.phase3_validation.validators import FunctionalValidator


def _artifact(content: str, tests: str) -> GeneratedArtifact:
    return GeneratedArtifact(template_id="t1", content=content, test_cases=tests)


def test_passing_test_executes_and_reports_coverage():
    validator = FunctionalValidator()
    result = validator.validate(_artifact("def f():\n    return 1", "def test_f():\n    assert f() == 1"))
    assert result.passed
    assert result.diagnostics["covered_lines"] >= 1
    assert result.diagnostics["tests_run"] == 1


def test_failing_assertion_fails_the_layer():
    validator = FunctionalValidator()
    result = validator.validate(_artifact("def f():\n    return 1", "def test_f():\n    assert f() == 2"))
    assert not result.passed
    assert result.layer == ValidationLayer.FUNCTIONAL
    assert "test_f" in result.details


def test_runtime_error_in_artifact_fails_the_layer():
    validator = FunctionalValidator()
    result = validator.validate(_artifact("undefined_name()", "def test_placeholder():\n    pass"))
    assert not result.passed
    assert "artifact setup" in result.details


def test_zero_coverage_fails_the_layer():
    """Tests that pass without ever calling into the generated code are a
    strong vacuous-test signal, not a real pass."""
    validator = FunctionalValidator()
    result = validator.validate(_artifact("def f():\n    return 1", "def test_f():\n    pass"))
    assert not result.passed
    assert "0% coverage" in result.details


def test_non_python_artifact_falls_back_to_presence_only():
    validator = FunctionalValidator()
    xtend_like = "class Foo extends Bar { def search(int limit) { limit } }"
    result = validator.validate(_artifact(xtend_like, "def test_search(): pass"))
    assert result.passed
    assert "not executable Python" in result.details


def test_junit_style_defers_to_production_toolchain():
    validator = FunctionalValidator()
    java_like = "public class FooTest { }"
    result = validator.validate(_artifact("public class Foo {}", java_like.replace("FooTest", "Foo") + "\n@Test void t() {}"))
    assert result.passed
    assert "JUnit" in result.details


def test_execution_timeout_fails_cleanly():
    validator = FunctionalValidator()
    validator.exec_timeout_seconds = 0.2
    infinite_loop = "def f():\n    while True:\n        pass"
    result = validator.validate(_artifact(infinite_loop, "def test_f():\n    f()"))
    assert not result.passed
    assert "timeout" in result.details

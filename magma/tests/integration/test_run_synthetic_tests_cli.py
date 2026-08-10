"""Locks in that the Synthetic Testing Agent is directly runnable, not
just importable -- magma/scripts/run_synthetic_tests.py had no test
before this."""
from __future__ import annotations

import json

from magma.scripts.run_synthetic_tests import main


def test_cli_runs_the_default_battery_and_all_scenarios_survive(capsys):
    exit_code = main(["--json"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code == 0
    assert report["survival_rate"] == 1.0
    assert len(report["outcomes"]) == 3
    assert all(o["survived"] for o in report["outcomes"])


def test_cli_human_readable_output(capsys):
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "survival rate: 100%" in captured.out
    assert "malformed_classification_output" in captured.out

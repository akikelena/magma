#!/usr/bin/env python3
"""CLI entry point for the Synthetic Testing Agent.

magma.phase4_improvement.synthetic_testing_agent.default_scenarios()
builds a real adversarial battery against real MAGMA agents (a
malformed LLM classification response, an empty change set, a
coherence check with nothing to sequence) but had no runnable entry
point anywhere in the repo. This script is that entry point: it
operates entirely offline on fresh, throwaway agent instances (a mock
LLM provider, an in-memory graph store), matching the agent's own
"operates offline on a copy of the configuration/pipeline components,
so it never mutates production state" contract -- there is nothing here
that touches a live Neo4j/Weaviate/Anthropic backend, deliberately.

Usage:
    python -m magma.scripts.run_synthetic_tests
    python -m magma.scripts.run_synthetic_tests --json
"""
from __future__ import annotations

import argparse
import json

from magma.knowledge_repository import InMemoryGraphStore
from magma.llm_providers import MockLLMProvider
from magma.phase1_documentation_detection.change_detection_agent import ChangeDetectionAgent
from magma.phase4_improvement.synthetic_testing_agent import SyntheticTestingAgent, default_scenarios


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print the full machine-readable report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # A deliberately unhelpful mock LLM response, so the
    # "malformed_classification_output" scenario actually exercises the
    # unparseable-response path rather than a scripted success.
    change_detection_agent = ChangeDetectionAgent(llm_provider=MockLLMProvider(responder=lambda p: "not valid json"))
    graph_store = InMemoryGraphStore()

    scenarios = default_scenarios(change_detection_agent, None, graph_store)
    result = SyntheticTestingAgent().process(scenarios)
    report = result.output

    if args.json:
        print(json.dumps({
            "survival_rate": report.survival_rate,
            "failure_patterns": report.failure_patterns,
            "outcomes": [{"name": o.name, "survived": o.survived, "description": o.description} for o in report.outcomes],
        }, indent=2))
    else:
        print(f"survival rate: {report.survival_rate:.0%}")
        for outcome in report.outcomes:
            status = "survived" if outcome.survived else "FAILED"
            print(f"  [{status}] {outcome.name}: {outcome.description}")

    return 0 if report.survival_rate == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

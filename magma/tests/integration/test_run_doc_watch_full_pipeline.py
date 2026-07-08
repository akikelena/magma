"""Proves --templates-config actually closes the loop from "documentation
changed" to "regenerated, validated template written back to disk" --
not just classification. Uses the same scripted responder and real
Milvus fixture pair as test_full_workflow.py (a known-good renamed-
parameter scenario), served over a local HTTP server since run_doc_watch
fetches over HTTP rather than reading local files."""
from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path

import pytest
import yaml

from magma.core.utils.config import MagmaConfig
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.llm_providers import MockLLMProvider
from magma.scripts.run_doc_watch import main

FIXTURES = Path(__file__).parent / "fixtures"


def _responder(prompt: str) -> str:
    """Identical to test_full_workflow.py's: a clean BREAKING rename
    (top_k -> limit) that regenerates and validates on the first attempt."""
    if "Reasoning from the previous stage" in prompt:
        return json.dumps({
            "template_content": "def search(limit):\n    return limit",
            "documentation_comments": "renamed top_k to limit",
            "test_cases": "def test_search():\n    assert search(5) == 5",
        })
    if "Audit trail:" in prompt:
        return "This episode renamed top_k to limit; validation passed on the first attempt."
    if prompt.strip().startswith("Fragment:"):
        return json.dumps({
            "change_type": "BREAKING", "confidence": 0.95, "intent_shift": 0.9,
            "compatibility_preserved": False, "affected_entities": ["top_k"], "rationale": "renamed",
        })
    return "reasoning: replace top_k with limit"


@pytest.fixture
def local_http_server():
    body_holder = {"body": (FIXTURES / "milvus_search_v2.3.md").read_bytes()}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.end_headers()
            self.wfile.write(body_holder["body"])

        def log_message(self, *args) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, body_holder
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_mapped_vendor_regenerates_validates_and_writes_the_template(tmp_path, capsys, local_http_server):
    pytest.importorskip("requests")
    server, body_holder = local_http_server
    port = server.server_address[1]

    template_path = tmp_path / "search_template.py"
    template_path.write_text("def search(top_k): ...")
    templates_config_path = tmp_path / "templates.yaml"
    templates_config_path.write_text(yaml.safe_dump({
        "local": [{"template_id": "search_template", "path": str(template_path), "heading": "Milvus Collection.search Reference/Parameters"}]
    }))

    config = MagmaConfig(vendor_sources={"local": {"url": f"http://127.0.0.1:{port}/docs"}})
    kr = KnowledgeRepositoryAgent()
    llm = MockLLMProvider(responder=_responder)

    # Baseline cycle: no prior snapshot exists yet, so this is also
    # reported as "changed" (there's nothing to diff against) and -- with
    # this content-blind scripted responder -- also regenerates. That's
    # expected: it's the second cycle below (a real v2.3 -> v2.4 diff)
    # that actually exercises the feature this test is for.
    main(
        ["--once", "--vendors", "local", "--json", "--templates-config", str(templates_config_path)],
        config=config, knowledge_repository=kr, llm_provider=llm,
    )
    capsys.readouterr()

    # Real change: v2.3 -> v2.4 renames top_k to limit.
    body_holder["body"] = (FIXTURES / "milvus_search_v2.4.md").read_bytes()
    exit_code = main(
        ["--once", "--vendors", "local", "--json", "--templates-config", str(templates_config_path)],
        config=config, knowledge_repository=kr, llm_provider=llm,
    )
    captured = capsys.readouterr()
    cycle = json.loads(captured.out)

    assert exit_code == 0
    entry = cycle["vendors"][0]
    assert entry["changed"] is True
    regeneration = entry["regeneration"]
    assert regeneration["status"] == "completed"
    assert regeneration["templates_validated_and_written"] == ["search_template"]
    assert regeneration.get("templates_kept_existing_content", []) == []

    # The actual point of the feature: the file on disk now has the
    # regenerated, validated content -- not the pre-change placeholder.
    assert template_path.read_text() == "def search(limit):\n    return limit"


def test_unmapped_vendor_is_unaffected_by_templates_config(tmp_path, capsys, local_http_server):
    """A vendor with no entry in --templates-config keeps the classify-
    only behavior -- --templates-config is additive, not a global switch
    that changes every vendor's handling."""
    pytest.importorskip("requests")
    server, body_holder = local_http_server
    port = server.server_address[1]

    templates_config_path = tmp_path / "templates.yaml"
    templates_config_path.write_text(yaml.safe_dump({
        "some-other-vendor": [{"template_id": "x", "path": str(tmp_path / "x.py"), "heading": "H"}]
    }))

    config = MagmaConfig(vendor_sources={"local": {"url": f"http://127.0.0.1:{port}/docs"}})
    kr = KnowledgeRepositoryAgent()
    llm = MockLLMProvider(responder=_responder)

    main(["--once", "--vendors", "local", "--json", "--templates-config", str(templates_config_path)], config=config, knowledge_repository=kr, llm_provider=llm)
    capsys.readouterr()

    body_holder["body"] = (FIXTURES / "milvus_search_v2.4.md").read_bytes()
    exit_code = main(["--once", "--vendors", "local", "--json", "--templates-config", str(templates_config_path)], config=config, knowledge_repository=kr, llm_provider=llm)
    captured = capsys.readouterr()
    cycle = json.loads(captured.out)

    assert exit_code == 0
    entry = cycle["vendors"][0]
    assert "classification" in entry
    assert "regeneration" not in entry


def test_failed_validation_keeps_the_existing_template_content(tmp_path, capsys, local_http_server):
    """MAGMA must never overwrite a working generator with an artifact
    it isn't confident in -- a regeneration whose validation fails must
    leave the on-disk file untouched."""
    pytest.importorskip("requests")
    server, body_holder = local_http_server
    port = server.server_address[1]

    def failing_responder(prompt: str) -> str:
        if "Reasoning from the previous stage" in prompt:
            # Deliberately broken Python -- fails Phase 3's syntax/execution layers.
            return json.dumps({"template_content": "def search(:\n", "documentation_comments": "", "test_cases": ""})
        if "Audit trail:" in prompt:
            return "regeneration failed validation"
        if prompt.strip().startswith("Fragment:"):
            return json.dumps({
                "change_type": "BREAKING", "confidence": 0.95, "intent_shift": 0.9,
                "compatibility_preserved": False, "affected_entities": ["top_k"], "rationale": "renamed",
            })
        return "reasoning: replace top_k with limit"

    template_path = tmp_path / "search_template.py"
    template_path.write_text("def search(top_k): ...")
    templates_config_path = tmp_path / "templates.yaml"
    templates_config_path.write_text(yaml.safe_dump({
        "local": [{"template_id": "search_template", "path": str(template_path), "heading": "Milvus Collection.search Reference/Parameters"}]
    }))

    config = MagmaConfig(vendor_sources={"local": {"url": f"http://127.0.0.1:{port}/docs"}})
    kr = KnowledgeRepositoryAgent()
    llm = MockLLMProvider(responder=failing_responder)

    main(["--once", "--vendors", "local", "--json", "--templates-config", str(templates_config_path)], config=config, knowledge_repository=kr, llm_provider=llm)
    capsys.readouterr()

    body_holder["body"] = (FIXTURES / "milvus_search_v2.4.md").read_bytes()
    main(["--once", "--vendors", "local", "--json", "--templates-config", str(templates_config_path)], config=config, knowledge_repository=kr, llm_provider=llm)
    captured = capsys.readouterr()
    cycle = json.loads(captured.out)

    entry = cycle["vendors"][0]
    regeneration = entry["regeneration"]
    assert "search_template" not in regeneration.get("templates_validated_and_written", [])
    assert template_path.read_text() == "def search(top_k): ..."  # untouched

"""Tests for DocumentWatcher's change-detection-over-time logic, using a
scripted fake DocumentFetcher (no network) -- consistent with how
MockLLMProvider stands in for a real LLM provider elsewhere in this
suite."""
from __future__ import annotations

from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.phase1_documentation_detection.document_source import DocumentFetcher, FetchResult, VendorDocSource
from magma.phase1_documentation_detection.document_watcher import DocumentWatcher


class _ScriptedFetcher(DocumentFetcher):
    """Returns each vendor's next content from a pre-scripted queue, so a
    test can simulate "the vendor's docs changed between checks" without
    any real HTTP call."""

    def __init__(self, contents_by_vendor: dict[str, list[str]]) -> None:
        self._queues = {vendor: list(contents) for vendor, contents in contents_by_vendor.items()}

    def fetch(self, source: VendorDocSource) -> FetchResult:
        content = self._queues[source.vendor].pop(0)
        return FetchResult(vendor=source.vendor, source_url=source.url, content=content)


def _sources() -> dict[str, VendorDocSource]:
    return {
        "acme": VendorDocSource(vendor="acme", url="https://example.com/acme/docs"),
        "globex": VendorDocSource(vendor="globex", url="https://example.com/globex/docs"),
    }


def test_first_check_is_always_reported_as_changed():
    watcher = DocumentWatcher(
        KnowledgeRepositoryAgent(), fetcher=_ScriptedFetcher({"acme": ["v1 content"]}), sources=_sources(),
    )
    result = watcher.check("acme")
    assert result.changed is True
    assert result.previous_content is None
    assert result.fetch.content == "v1 content"


def test_second_check_with_identical_content_is_not_changed():
    fetcher = _ScriptedFetcher({"acme": ["same content", "same content"]})
    watcher = DocumentWatcher(KnowledgeRepositoryAgent(), fetcher=fetcher, sources=_sources())

    first = watcher.check("acme")
    second = watcher.check("acme")

    assert first.changed is True
    assert second.changed is False
    assert second.previous_content == "same content"


def test_second_check_with_different_content_is_changed():
    fetcher = _ScriptedFetcher({"acme": ["old content", "new content"]})
    watcher = DocumentWatcher(KnowledgeRepositoryAgent(), fetcher=fetcher, sources=_sources())

    watcher.check("acme")
    second = watcher.check("acme")

    assert second.changed is True
    assert second.previous_content == "old content"
    assert second.fetch.content == "new content"


def test_last_known_returns_none_before_any_check():
    watcher = DocumentWatcher(KnowledgeRepositoryAgent(), fetcher=_ScriptedFetcher({}), sources=_sources())
    assert watcher.last_known("acme") is None


def test_last_known_reflects_the_most_recent_fetch():
    fetcher = _ScriptedFetcher({"acme": ["v1", "v2"]})
    watcher = DocumentWatcher(KnowledgeRepositoryAgent(), fetcher=fetcher, sources=_sources())

    watcher.check("acme")
    watcher.check("acme")

    assert watcher.last_known("acme").content == "v2"


def test_check_unknown_vendor_raises_key_error():
    watcher = DocumentWatcher(KnowledgeRepositoryAgent(), fetcher=_ScriptedFetcher({}), sources=_sources())
    try:
        watcher.check("not_a_real_vendor")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_check_all_covers_every_registered_vendor_in_sorted_order():
    fetcher = _ScriptedFetcher({"acme": ["a-content"], "globex": ["g-content"]})
    watcher = DocumentWatcher(KnowledgeRepositoryAgent(), fetcher=fetcher, sources=_sources())

    results = watcher.check_all()

    assert [r.vendor for r in results] == ["acme", "globex"]
    assert all(r.changed for r in results)


def test_snapshots_are_versioned_and_survive_across_watcher_instances():
    """Persistence goes through KnowledgeRepositoryAgent, not watcher-local
    state -- a fresh DocumentWatcher over the same repository must see the
    prior snapshot."""
    kr = KnowledgeRepositoryAgent()
    fetcher = _ScriptedFetcher({"acme": ["v1", "v2"]})
    DocumentWatcher(kr, fetcher=fetcher, sources=_sources()).check("acme")

    second_watcher = DocumentWatcher(kr, fetcher=_ScriptedFetcher({"acme": ["v2"]}), sources=_sources())
    result = second_watcher.check("acme")

    assert result.previous_content == "v1"
    assert result.changed is True

    history = kr.history("document_watcher::acme")
    assert len(history) == 2


def test_check_all_isolates_one_vendors_failure_from_the_rest():
    class _OneFailsFetcher(DocumentFetcher):
        def fetch(self, source: VendorDocSource) -> FetchResult:
            if source.vendor == "acme":
                raise ConnectionError("simulated network failure")
            return FetchResult(vendor=source.vendor, source_url=source.url, content="fine")

    watcher = DocumentWatcher(KnowledgeRepositoryAgent(), fetcher=_OneFailsFetcher(), sources=_sources())

    results = watcher.check_all()

    assert [r.vendor for r in results] == ["globex"]
    assert "acme" in watcher.last_errors
    assert "simulated network failure" in watcher.last_errors["acme"]


def test_check_all_clears_previous_errors_on_a_later_successful_run():
    calls = {"n": 0}

    class _FailsOnceFetcher(DocumentFetcher):
        def fetch(self, source: VendorDocSource) -> FetchResult:
            calls["n"] += 1
            if source.vendor == "acme" and calls["n"] <= 1:
                raise ConnectionError("first attempt fails")
            return FetchResult(vendor=source.vendor, source_url=source.url, content="fine")

    watcher = DocumentWatcher(KnowledgeRepositoryAgent(), fetcher=_FailsOnceFetcher(), sources=_sources())

    watcher.check_all()
    assert "acme" in watcher.last_errors

    watcher.check_all()
    assert watcher.last_errors == {}

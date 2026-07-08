"""Tests for the real vendor documentation fetcher.

Two layers, matching the pattern used for the other real-library-backed
modules in this repo (neo4j/weaviate): fast unit tests against a fake
`requests` module for retry/error-handling logic, and a real-library test
using `pytest.importorskip("requests")` against a local `http.server`
instance so the real `requests.get(...)` call path is exercised without
depending on live internet access from CI.
"""
from __future__ import annotations

import http.server
import sys
import threading
import types

import pytest

from magma.core.utils.config import MagmaConfig
from magma.phase1_documentation_detection.document_source import (
    DEFAULT_VENDOR_DOC_SOURCES,
    DocumentFetchError,
    FetchResult,
    HttpDocumentFetcher,
    VendorDocSource,
    vendor_doc_sources,
)


def test_default_vendor_registry_covers_the_five_evaluated_vdbmss():
    assert set(DEFAULT_VENDOR_DOC_SOURCES) == {"milvus", "weaviate", "pinecone", "qdrant", "chroma"}
    for source in DEFAULT_VENDOR_DOC_SOURCES.values():
        assert source.url.startswith("https://")


def test_qdrant_is_flagged_as_requiring_js_rendering():
    # Verified directly: a plain HTTP GET against Qdrant's live docs site
    # returns an empty page shell, not the article body -- this flag is
    # what stops HttpDocumentFetcher from silently trusting that as real
    # content.
    assert DEFAULT_VENDOR_DOC_SOURCES["qdrant"].requires_js_rendering is True
    assert DEFAULT_VENDOR_DOC_SOURCES["milvus"].requires_js_rendering is False


def test_vendor_doc_sources_merges_config_over_defaults():
    config = MagmaConfig(vendor_sources={"milvus": {"url": "https://example.com/milvus-mirror.md"}})
    sources = vendor_doc_sources(config)
    assert sources["milvus"].url == "https://example.com/milvus-mirror.md"
    # untouched vendors keep the library default
    assert sources["weaviate"] == DEFAULT_VENDOR_DOC_SOURCES["weaviate"]


def test_fetch_result_computes_its_own_content_hash():
    result = FetchResult(vendor="v", source_url="u", content="hello world")
    assert len(result.content_hash) == 64  # sha256 hex digest
    assert FetchResult(vendor="v", source_url="u", content="hello world").content_hash == result.content_hash
    assert FetchResult(vendor="v", source_url="u", content="different").content_hash != result.content_hash


# -- Fake-requests unit tests: retry/backoff/error-handling logic ---------

class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200, content_type: str = "text/markdown"):
        self.text = text
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise self._FakeHTTPError(f"HTTP {self.status_code}")

    class _FakeHTTPError(Exception):
        pass


@pytest.fixture
def fake_requests(monkeypatch):
    fake = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    fake.RequestException = RequestException
    fake.get = None  # set per-test
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


def test_fetch_retries_transient_failures_then_succeeds(fake_requests, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    calls = {"n": 0}

    def flaky_get(url, timeout, headers):
        calls["n"] += 1
        if calls["n"] < 3:
            raise fake_requests.RequestException("connection reset")
        return _FakeResponse("# Real content\n\n" + "Enough text to pass the trust threshold check comfortably here. " * 5)

    fake_requests.get = flaky_get
    fetcher = HttpDocumentFetcher(max_attempts=3, backoff_seconds=0)
    source = VendorDocSource(vendor="v", url="https://example.com/docs")

    result = fetcher.fetch(source)

    assert calls["n"] == 3
    assert "Real content" in result.content
    assert result.is_placeholder is False


def test_fetch_raises_document_fetch_error_after_exhausting_retries(fake_requests, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    def always_fails(url, timeout, headers):
        raise fake_requests.RequestException("connection refused")

    fake_requests.get = always_fails
    fetcher = HttpDocumentFetcher(max_attempts=2, backoff_seconds=0)
    source = VendorDocSource(vendor="v", url="https://example.com/docs")

    with pytest.raises(DocumentFetchError, match="after 2 attempts"):
        fetcher.fetch(source)


def test_fetch_strips_html_to_text_when_content_type_is_html(fake_requests):
    html = "<html><head><style>.x{}</style></head><body><script>evil()</script><h1>Title</h1><p>" + ("Body text. " * 20) + "</p></body></html>"
    fake_requests.get = lambda url, timeout, headers: _FakeResponse(html, content_type="text/html; charset=utf-8")
    fetcher = HttpDocumentFetcher(max_attempts=1)

    result = fetcher.fetch(VendorDocSource(vendor="v", url="https://example.com/docs"))

    assert "evil()" not in result.content
    assert ".x{}" not in result.content
    assert "Title" in result.content
    assert "Body text." in result.content


def test_short_or_empty_fetch_is_flagged_as_placeholder(fake_requests):
    fake_requests.get = lambda url, timeout, headers: _FakeResponse("")
    fetcher = HttpDocumentFetcher(max_attempts=1)

    result = fetcher.fetch(VendorDocSource(vendor="v", url="https://example.com/docs"))

    assert result.is_placeholder is True


def test_requires_js_rendering_vendor_is_always_flagged_even_with_long_content(fake_requests):
    fake_requests.get = lambda url, timeout, headers: _FakeResponse("word " * 500)
    fetcher = HttpDocumentFetcher(max_attempts=1)
    source = VendorDocSource(vendor="qdrant", url="https://example.com/docs", requires_js_rendering=True)

    result = fetcher.fetch(source)

    assert result.is_placeholder is True


# -- Real requests library against a local HTTP server --------------------

@pytest.fixture
def local_http_server():
    body_holder = {"body": b"# Placeholder\n", "content_type": "text/markdown"}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
            self.send_response(200)
            self.send_header("Content-Type", body_holder["content_type"])
            self.end_headers()
            self.wfile.write(body_holder["body"])

        def log_message(self, *args) -> None:  # silence default request logging
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, body_holder
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_real_requests_library_fetches_markdown_from_a_local_server(local_http_server):
    pytest.importorskip("requests")
    server, body_holder = local_http_server
    port = server.server_address[1]
    body_holder["body"] = ("# Real Markdown\n\n" + "Search reference content. " * 20).encode("utf-8")
    body_holder["content_type"] = "text/markdown; charset=utf-8"

    fetcher = HttpDocumentFetcher(timeout_seconds=5, max_attempts=1)
    source = VendorDocSource(vendor="local", url=f"http://127.0.0.1:{port}/docs")

    result = fetcher.fetch(source)

    assert "Real Markdown" in result.content
    assert result.is_placeholder is False
    assert result.vendor == "local"


def test_real_requests_library_strips_html_from_a_local_server(local_http_server):
    pytest.importorskip("requests")
    server, body_holder = local_http_server
    port = server.server_address[1]
    body_holder["body"] = (
        b"<html><head><style>.x{color:red}</style></head><body>"
        b"<script>doEvil()</script><h1>Search</h1><p>"
        + b"Real paragraph content. " * 20
        + b"</p></body></html>"
    )
    body_holder["content_type"] = "text/html; charset=utf-8"

    fetcher = HttpDocumentFetcher(timeout_seconds=5, max_attempts=1)
    result = fetcher.fetch(VendorDocSource(vendor="local", url=f"http://127.0.0.1:{port}/docs"))

    assert "doEvil()" not in result.content
    assert "color:red" not in result.content
    assert "Search" in result.content
    assert "Real paragraph content." in result.content

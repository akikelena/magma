"""Real documentation acquisition for the Documentation Processing Agent.

The agent's own contract takes "raw documentation retrieved from HTTP
endpoints, version-controlled repositories, or APIs" as a precondition --
this module is that acquisition mechanism for the HTTP case: a registry of
the vendor documentation entry points MAGMA's published evaluation
targets (Milvus, Weaviate, Pinecone, Qdrant, Chroma), and a real fetcher
that retrieves and normalizes their content into plain text a
`RawDocument` can be built from.

`DocumentFetcher` is a small, storage-agnostic interface (mirroring
`GraphStore`/`VectorStore`/`LLMProvider`) so the default `HttpDocumentFetcher`
can be substituted for a browser-rendering backend where a vendor's docs
site is a client-rendered SPA (see `VendorDocSource.requires_js_rendering`).
"""
from __future__ import annotations

import abc
import hashlib
import html.parser
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from magma.core.utils.config import MagmaConfig, load_config

_DEFAULT_TIMEOUT_SECONDS = 10
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BACKOFF_SECONDS = 1.0
# Below this many characters, a "successful" HTTP fetch is treated as
# suspect rather than trusted: client-rendered SPA shells and error pages
# both tend to be short, and silently diffing against a near-empty
# document would produce a false "everything was deleted" change.
_MIN_TRUSTED_CONTENT_CHARS = 200


@dataclass(frozen=True)
class VendorDocSource:
    vendor: str
    url: str
    description: str = ""
    # True when the page is known to be client-rendered (confirmed for
    # Qdrant: a plain HTTP GET returns an empty shell, not the article
    # body). HttpDocumentFetcher still attempts the fetch rather than
    # refusing outright -- vendors sometimes serve a server-rendered
    # fallback to non-browser clients -- but the result is flagged so
    # callers don't treat an empty scrape as "no change since last time."
    requires_js_rendering: bool = False


# Library/test default, independent of config/vendor_sources.yaml being
# loaded -- mirrors config/vendor_sources.yaml exactly; keep the two in
# sync (test_document_source.py locks this in).
DEFAULT_VENDOR_DOC_SOURCES: dict[str, VendorDocSource] = {
    "milvus": VendorDocSource(
        vendor="milvus", url="https://milvus.io/docs/single-vector-search.md",
        description="Milvus ANN search reference",
    ),
    "weaviate": VendorDocSource(
        vendor="weaviate", url="https://docs.weaviate.io/weaviate/search/similarity",
        description="Weaviate similarity search reference",
    ),
    "pinecone": VendorDocSource(
        vendor="pinecone", url="https://docs.pinecone.io/guides/search/search-overview",
        description="Pinecone search overview reference",
    ),
    "qdrant": VendorDocSource(
        vendor="qdrant", url="https://qdrant.tech/documentation/concepts/search/",
        description="Qdrant search concepts reference", requires_js_rendering=True,
    ),
    "chroma": VendorDocSource(
        vendor="chroma", url="https://docs.trychroma.com/docs/querying-collections/query-and-get",
        description="Chroma query/get reference",
    ),
}


def vendor_doc_sources(config: MagmaConfig | None = None) -> dict[str, VendorDocSource]:
    """Vendor registry sourced from config/vendor_sources.yaml, falling
    back to `DEFAULT_VENDOR_DOC_SOURCES` entry-by-entry for any vendor the
    config doesn't mention (so a partial override file is enough)."""
    config = config or load_config()
    sources = dict(DEFAULT_VENDOR_DOC_SOURCES)
    for vendor, raw in config.vendor_sources.items():
        sources[vendor] = VendorDocSource(
            vendor=vendor,
            url=raw["url"],
            description=raw.get("description", ""),
            requires_js_rendering=bool(raw.get("requires_js_rendering", False)),
        )
    return sources


@dataclass(frozen=True)
class FetchResult:
    vendor: str
    source_url: str
    content: str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str = ""
    is_placeholder: bool = False

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", hashlib.sha256(self.content.encode("utf-8")).hexdigest())


class DocumentFetchError(Exception):
    """Raised when a fetch exhausts its retry budget without succeeding."""


class DocumentFetcher(abc.ABC):
    @abc.abstractmethod
    def fetch(self, source: VendorDocSource) -> FetchResult: ...


class _TextExtractingHTMLParser(html.parser.HTMLParser):
    """Strips markup down to visible text: drops script/style contents and
    collapses tag boundaries to whitespace. Deliberately simple -- this is
    a fallback for vendors that serve rendered HTML (e.g. Weaviate) rather
    than raw markdown (e.g. Milvus/Pinecone/Chroma's `.md`-suffixed docs,
    which need no HTML stripping at all)."""

    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())

    def text(self) -> str:
        return "\n".join(self._chunks)


def _html_to_text(raw: str) -> str:
    parser = _TextExtractingHTMLParser()
    parser.feed(raw)
    return parser.text()


class HttpDocumentFetcher(DocumentFetcher):
    """Real HTTP(S) fetcher: GETs `source.url`, retries transient failures
    with exponential backoff, and normalizes the response into plain text
    -- markdown/plain-text responses (Milvus/Pinecone/Chroma's docs sites
    all serve these) pass through unchanged, HTML responses (Weaviate) are
    stripped to visible text via `_html_to_text`.

    `requests` is imported lazily, matching every other optional backend
    in this repo -- constructing this class works without it installed;
    only `fetch()` needs it.
    """

    def __init__(
        self,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = _DEFAULT_BACKOFF_SECONDS,
        user_agent: str = "MAGMA-DocumentWatcher/1.0 (+https://github.com/)",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.user_agent = user_agent

    def fetch(self, source: VendorDocSource) -> FetchResult:
        import requests  # noqa: F401 (lazy import)

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = requests.get(
                    source.url, timeout=self.timeout_seconds,
                    headers={"User-Agent": self.user_agent, "Accept": "text/markdown, text/html, text/plain"},
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:  # noqa: PERF203 (retry loop, not a hot path)
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        else:
            raise DocumentFetchError(
                f"failed to fetch {source.vendor!r} documentation from {source.url!r} "
                f"after {self.max_attempts} attempts: {last_error!r}"
            )

        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type:
            text = _html_to_text(response.text)
        else:
            text = response.text

        is_placeholder = source.requires_js_rendering or len(text.strip()) < _MIN_TRUSTED_CONTENT_CHARS
        return FetchResult(vendor=source.vendor, source_url=source.url, content=text, is_placeholder=is_placeholder)

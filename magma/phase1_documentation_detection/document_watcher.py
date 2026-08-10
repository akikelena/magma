"""Change-aware documentation acquisition: fetches a vendor's live docs,
compares against the last snapshot MAGMA stored for it, and persists the
new snapshot -- the deterministic core behind "detect changes in docs" as
a standalone, schedulable operation, independent of whether the caller
goes on to run full Phase 1 classification.

Persistence reuses `KnowledgeRepositoryAgent`'s `MessageBus` interface
(`write`/`read`/`history`) rather than a new storage mechanism: each
vendor's fetched content is a versioned blackboard entry keyed by vendor,
so the existing rollback/history machinery applies to it for free, and no
new backend is introduced for what is, structurally, just another piece
of blackboard state.

Deliberately deterministic and CrewAI-independent: this is the routing
question of "did anything change," not a semantic-judgment question, so
it stays a plain Python service the CrewAI layer calls as a tool rather
than something an LLM decides how to do.
"""
from __future__ import annotations

from dataclasses import dataclass

from magma.core.governance.audit_log import AuditLogger
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.phase1_documentation_detection.document_source import (
    DocumentFetcher,
    FetchResult,
    HttpDocumentFetcher,
    VendorDocSource,
    vendor_doc_sources,
)

_STORE_KEY_PREFIX = "document_watcher::"


@dataclass(frozen=True)
class DocumentWatcherResult:
    vendor: str
    source_url: str
    changed: bool
    fetch: FetchResult
    previous_content: str | None
    previous_version_id: str | None
    new_version_id: str


class DocumentWatcher:
    """Checks one or more vendors' live documentation against the last
    snapshot stored in the Knowledge Repository, persisting a new
    snapshot whenever the content actually changed."""

    def __init__(
        self,
        knowledge_repository: KnowledgeRepositoryAgent,
        fetcher: DocumentFetcher | None = None,
        sources: dict[str, VendorDocSource] | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.knowledge_repository = knowledge_repository
        self.fetcher = fetcher or HttpDocumentFetcher()
        self.sources = sources if sources is not None else vendor_doc_sources()
        self.audit = audit_logger or AuditLogger()
        # Populated by check_all(): vendor -> error message, for vendors
        # whose fetch failed outright (network error, exhausted retries).
        # Mirrors DocumentationProcessingAgent.skipped -- a failure for
        # one source is recorded and skipped rather than allowed to take
        # down a batch check of every other vendor.
        self.last_errors: dict[str, str] = {}

    @staticmethod
    def _store_key(vendor: str) -> str:
        return f"{_STORE_KEY_PREFIX}{vendor}"

    def last_known(self, vendor: str) -> FetchResult | None:
        """The most recently stored fetch for `vendor`, or None if this
        vendor has never been checked before."""
        return self.knowledge_repository.read(self._store_key(vendor))

    def check(self, vendor: str) -> DocumentWatcherResult:
        """Fetch `vendor`'s current documentation and compare it against
        the last stored snapshot. Always persists the new snapshot (even
        when unchanged, so `fetched_at` reflects the last check time);
        `changed` is True whenever the content hash differs from the
        previous snapshot, or there was no previous snapshot at all."""
        if vendor not in self.sources:
            raise KeyError(f"no documentation source registered for vendor {vendor!r}")
        source = self.sources[vendor]

        previous = self.last_known(vendor)
        fetch = self.fetcher.fetch(source)
        changed = previous is None or previous.content_hash != fetch.content_hash

        key = self._store_key(vendor)
        new_version_id = self.knowledge_repository.write(key, fetch)
        self.audit.log(
            "DocumentWatcher", "check", episode_id=key,
            vendor=vendor, changed=changed, is_placeholder=fetch.is_placeholder,
            content_hash=fetch.content_hash[:16],
        )
        return DocumentWatcherResult(
            vendor=vendor, source_url=source.url, changed=changed, fetch=fetch,
            previous_content=previous.content if previous is not None else None,
            previous_version_id=self.knowledge_repository.history(key)[-2] if len(self.knowledge_repository.history(key)) > 1 else None,
            new_version_id=new_version_id,
        )

    def check_all(self) -> list[DocumentWatcherResult]:
        """`check()` every registered vendor, in a stable (sorted) order
        so results and logs are reproducible across runs.

        A vendor whose fetch fails outright (network error, exhausted
        retries) is recorded in `last_errors` and skipped rather than
        raised -- one vendor's outage must not stop the rest of a
        scheduled multi-vendor check from completing.
        """
        self.last_errors = {}
        results: list[DocumentWatcherResult] = []
        for vendor in sorted(self.sources):
            try:
                results.append(self.check(vendor))
            except Exception as exc:  # noqa: BLE001 - isolate one vendor's failure from the batch
                self.last_errors[vendor] = str(exc)
                self.audit.log(
                    "DocumentWatcher", "check_failed", episode_id=self._store_key(vendor),
                    vendor=vendor, error=str(exc),
                )
        return results

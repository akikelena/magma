"""Append-only audit logging.

Mandatory governance requires that every agent action is logged, so that
the Explainability Agent (Phase 4) can reconstruct the reasoning path from
documentation change through template update to deployed generator. This
module is the write side of that trail; `AuditLogger` is deliberately
storage-agnostic (default in-memory + JSONL sink) so it can later be backed
by the Knowledge Repository without changing call sites.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditRecord:
    agent_name: str
    action: str
    episode_id: str
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Collects `AuditRecord`s in memory and optionally persists them as JSONL."""

    def __init__(self, sink_path: str | Path | None = None) -> None:
        self._records: list[AuditRecord] = []
        self._sink_path = Path(sink_path) if sink_path else None

    def log(self, agent_name: str, action: str, episode_id: str, **details: Any) -> AuditRecord:
        record = AuditRecord(agent_name=agent_name, action=action, episode_id=episode_id, details=details)
        self._records.append(record)
        if self._sink_path is not None:
            with self._sink_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        return record

    def for_episode(self, episode_id: str) -> list[AuditRecord]:
        return [r for r in self._records if r.episode_id == episode_id]

    def all(self) -> list[AuditRecord]:
        return list(self._records)

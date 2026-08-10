"""MessageBus / blackboard access interface.

MAGMA is a blackboard architecture (Hayes-Roth): components do not address
one another directly, they read state from a shared, versioned repository
and write their outputs back to it. `MessageBus` is the narrow interface
every agent depends on for that read/write access, so that the concrete
storage (Neo4j + Weaviate, see `magma.knowledge_repository`) stays an
implementation detail behind the Knowledge Repository Agent.
"""
from __future__ import annotations

import abc
from typing import Any, Iterable


class MessageBus(abc.ABC):
    """Narrow read/write contract over the shared knowledge repository."""

    @abc.abstractmethod
    def write(self, key: str, value: Any, *, version: bool = True) -> str:
        """Persist `value` under `key`, returning a version/revision id."""
        raise NotImplementedError

    @abc.abstractmethod
    def read(self, key: str, *, version: str | None = None) -> Any:
        """Read the latest (or a specific historical) value for `key`."""
        raise NotImplementedError

    @abc.abstractmethod
    def history(self, key: str) -> Iterable[str]:
        """Return version ids for `key`, oldest first."""
        raise NotImplementedError

    @abc.abstractmethod
    def rollback(self, key: str, version: str) -> None:
        """Restore `key` to a previously persisted version."""
        raise NotImplementedError

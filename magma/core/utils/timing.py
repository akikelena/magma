"""Wall-clock timing helper for audit-log latency tracking.

`UsageFeedbackAgent` needs to identify the slowest stages, which requires
actual elapsed time, not just how often an action fires. `timed()` is a
small context manager agents wrap around their LLM-calling (or otherwise
slow) work; the elapsed milliseconds are then passed as `duration_ms`
into `AuditLogger.log`, where they land in `AuditRecord.details` for
`UsageFeedbackAgent` to aggregate.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class _Elapsed:
    def __init__(self) -> None:
        self.ms: float = 0.0


@contextmanager
def timed() -> Iterator[_Elapsed]:
    """Usage:

        with timed() as elapsed:
            do_slow_thing()
        audit.log(..., duration_ms=elapsed.ms)
    """
    box = _Elapsed()
    start = time.perf_counter()
    try:
        yield box
    finally:
        box.ms = (time.perf_counter() - start) * 1000.0

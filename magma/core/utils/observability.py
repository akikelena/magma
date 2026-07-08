"""Structured logging helper.

Provides one consistently-configured logger factory so every agent's log
lines share a format the Usage Feedback Agent can parse. Metrics export
(Prometheus) lives in `magma.core.utils.metrics` and is wired at the
application boundary in `magma.orchestration.MagmaWorkflow`, not here, to
keep this module dependency-light.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s :: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root = logging.getLogger("magma")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(f"magma.{name}")

"""Sahayak AI — shared logging configuration.

Call configure_logging() once, at application startup (backend/main.py),
before anything else runs. All modules should then use:

    import logging
    logger = logging.getLogger(__name__)

instead of print(), so output has timestamps, levels, and can be routed
to files/log aggregators later without touching call sites.
"""
from __future__ import annotations

import logging
import os
import sys


def configure_logging(level: int | None = None) -> None:
    """Configure root logging once for the whole app.

    Level can be overridden via the LOG_LEVEL env var (e.g. DEBUG, WARNING).
    Defaults to INFO.
    """
    if level is None:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # override any default handlers set up by imported libraries
    )

    # Quiet down noisy third-party loggers unless we're debugging.
    if level > logging.DEBUG:
        for noisy in ("httpx", "urllib3", "sentence_transformers", "faiss"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
"""Centralised logging configuration.

A single rotating logger is written to ``<project_root>/logs/phs.log``
*and* echoed to stderr during GUI-less runs. The GUI can attach its own
handler to forward records to a docked console view later.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "plant_hsi"


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    """Initialise the application root logger and return it."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(
        log_dir / "phs.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger, creating the root lazily if needed."""
    if not logging.getLogger(_LOGGER_NAME).handlers:
        setup_logging(Path.home() / ".config" / "plant_hsi" / "logs")
    return logging.getLogger(_LOGGER_NAME).getChild(name)

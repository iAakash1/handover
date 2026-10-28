"""Structured logging setup.

Configures the ``plantdx`` logger with a console handler and, optionally, a file
handler (used by the audit engine to write ``reports/audit.log``). Uses ``rich``
for pretty console output when available, otherwise the standard stream handler.
"""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _console_handler(rich: bool) -> logging.Handler:
    """Return a console handler, preferring ``rich`` if installed."""
    if rich:
        try:
            from rich.logging import RichHandler

            return RichHandler(rich_tracebacks=True, show_path=False)
        except ImportError:
            pass
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    return handler


def configure_logging(
    level: str = "INFO",
    *,
    rich: bool = True,
    log_file: str | Path | None = None,
) -> None:
    """Configure the root ``plantdx`` logger.

    Args:
        level: Logging level name (``DEBUG``/``INFO``/``WARNING``/``ERROR``).
        rich: Use the ``rich`` console handler when available.
        log_file: If given, also write a plain-text log to this file (mode ``w``).
    """
    root = logging.getLogger("plantdx")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False
    root.addHandler(_console_handler(rich))
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger under ``plantdx``."""
    return logging.getLogger(name if name.startswith("plantdx") else f"plantdx.{name}")

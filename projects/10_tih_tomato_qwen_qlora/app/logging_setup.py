"""A single file logger for the demo, for debugging predictions and loads.

Writes timestamped lines to ``logs/plantdx_app.log``. Configured once; safe to
call ``get_logger()`` from anywhere. Never raises if the log directory can't be
created — logging must not break the app.
"""

from __future__ import annotations

import logging
from logging import Logger

from app.utils import APP_LOG, LOGS_DIR

_LOGGER_NAME = "plantdx.app"
_configured = False


def get_logger() -> Logger:
    """Return the app logger, configuring a file handler exactly once."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(APP_LOG, encoding="utf-8")
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    _configured = True
    return logger

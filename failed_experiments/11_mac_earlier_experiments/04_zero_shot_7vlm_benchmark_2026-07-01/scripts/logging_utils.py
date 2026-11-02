"""
Per-model logging.

Each model gets its own named logger writing to both the console (same
plain, unadorned messages as before) and a persistent
logs/<model_slug>.log file (timestamped, leveled), so download/load
status, per-image progress, warnings, and errors for a given model are
all recorded somewhere that survives after the terminal is closed.
"""

import logging
from pathlib import Path

_configured_loggers: dict = {}


def get_model_logger(model_slug: str, logs_dir: Path) -> logging.Logger:
    """
    Return a logger named "benchmark.<model_slug>" configured to write to
    both stdout and logs/<model_slug>.log. Safe to call more than once
    for the same model_slug within a process — handlers are only attached
    the first time, so repeated calls (e.g. across --force re-runs in the
    same invocation) never duplicate log lines.
    """
    if model_slug in _configured_loggers:
        return _configured_loggers[model_slug]

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"benchmark.{model_slug}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # keep each model's records out of each other's file

    file_handler = logging.FileHandler(logs_dir / f"{model_slug}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))  # plain, matches prior console UX
    logger.addHandler(console_handler)

    _configured_loggers[model_slug] = logger
    return logger


def get_run_logger(logs_dir: Path) -> logging.Logger:
    """Logger for run-level (not model-specific) messages, e.g. dataset
    discovery, aggregation, validation, and the final summary. Writes to
    logs/run.log plus the console."""
    return get_model_logger("run", logs_dir)

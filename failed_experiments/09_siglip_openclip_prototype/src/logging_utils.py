"""Logging utilities and experiment run directory management."""

import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import yaml

from src.config import ConfigDict
from src.utils import ensure_dir


def create_run_directory(base_output_dir: str = "outputs/runs", experiment_name: str = "experiment") -> Path:
    """
    Create a unique timestamped experiment run directory.
    Example: outputs/runs/2026-08-27_120000_openclip_vitb16/
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    clean_name = experiment_name.replace(" ", "_").replace("/", "_")
    run_dir = Path(base_output_dir) / f"{timestamp}_{clean_name}"
    
    # Subdirectories
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "checkpoints" / "periodic")
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "plots")
    ensure_dir(run_dir / "results")
    ensure_dir(run_dir / "prototypes")
    
    return run_dir


def setup_logger(
    name: str = "tomato_vlm",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configure formatted logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_run_config(config: ConfigDict, run_dir: Path) -> Path:
    """Save an exact copy of the configuration in the experiment directory."""
    config_path = run_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    return config_path

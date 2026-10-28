"""Utility functions for file I/O, atomic checkpointing, and metrics serialization."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Union
import torch


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_save_torch(obj: Any, target_path: Union[str, Path]) -> None:
    """
    Safely save a PyTorch object atomically.
    Saves to a temporary file in the same directory and replaces target_path.
    Prevents file corruption if process is killed or interrupted.
    """
    target = Path(target_path).resolve()
    ensure_dir(target.parent)
    
    # Create temp file in the same folder so atomic rename (os.replace) works across filesystems
    temp_file = tempfile.NamedTemporaryFile(
        dir=str(target.parent),
        prefix=f".tmp_{target.name}_",
        delete=False
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, target)
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise RuntimeError(f"Failed to atomically save torch file to {target}: {e}") from e


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file with atomic write."""
    target = Path(filepath).resolve()
    ensure_dir(target.parent)
    temp_path = target.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
        os.replace(temp_path, target)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to save JSON to {target}: {e}") from e


def load_json(filepath: Union[str, Path]) -> Any:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> Dict[str, Union[int, float]]:
    """Count total, trainable, and frozen parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_percentage": round(trainable_pct, 4)
    }


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

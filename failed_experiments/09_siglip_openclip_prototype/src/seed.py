"""Seed management for reproducibility and environment metadata tracking."""

import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

from src.utils import save_json


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_environment_info(seed: int = 42, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Capture complete software, hardware, and environment metadata."""
    # Package versions
    packages = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }

    try:
        import torchvision
        packages["torchvision_version"] = torchvision.__version__
    except ImportError:
        packages["torchvision_version"] = "not_installed"

    try:
        import transformers
        packages["transformers_version"] = transformers.__version__
    except ImportError:
        packages["transformers_version"] = "not_installed"

    try:
        import open_clip
        packages["open_clip_version"] = getattr(open_clip, "__version__", "installed")
    except ImportError:
        packages["open_clip_version"] = "not_installed"

    # Hardware metadata
    hardware: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }

    info: Dict[str, Any] = {
        "random_seed": seed,
        "packages": packages,
        "hardware": hardware,
    }

    if model_config:
        info["model_configuration"] = model_config

    return info


def save_environment_info(filepath: Path, seed: int = 42, model_config: Optional[Dict[str, Any]] = None) -> None:
    """Save environment metadata to environment.json."""
    info = get_environment_info(seed, model_config)
    save_json(info, filepath)

"""Hardware and device management with automatic CUDA, MPS, and CPU detection."""

import os
from typing import Tuple, Optional
import torch


def get_device(preferred: str = "auto") -> torch.device:
    """
    Detect and return the optimal compute device according to priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (Fallback)
    
    If 'preferred' is specified and valid, respects the preference.
    """
    pref = (preferred or "auto").lower()

    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        raise RuntimeError("CUDA was explicitly requested, but torch.cuda.is_available() is False.")

    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was explicitly requested, but Apple Silicon MPS is not available.")

    if pref == "cpu":
        return torch.device("cpu")

    # Auto detection priority: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def setup_hardware_optimizations(device: torch.device, enable_tf32: bool = True) -> None:
    """Configure device-specific performance optimizations."""
    if device.type == "cuda":
        if enable_tf32 and hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def get_autocast_context(device: torch.device, mixed_precision: str = "auto") -> Tuple[bool, torch.dtype, str]:
    """
    Determine if AMP is supported for device and return (enabled, dtype, device_type).
    """
    mode = (mixed_precision or "auto").lower()

    if mode in ("no", "none", "false"):
        return False, torch.float32, device.type

    if device.type == "cuda":
        # Check if bfloat16 is supported (Ampere or newer, compute capability >= 8.0)
        if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
            return True, torch.bfloat16, "cuda"
        # Otherwise use standard fp16
        return True, torch.float16, "cuda"

    if device.type == "mps":
        # MPS autocast support is limited / experimental across versions
        if mode in ("fp16", "auto") and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                # Test autocast on mps
                with torch.amp.autocast(device_type="mps", dtype=torch.float16):
                    _ = torch.zeros(1, device="mps")
                return True, torch.float16, "mps"
            except Exception:
                return False, torch.float32, "mps"
        return False, torch.float32, "mps"

    # CPU
    if mode == "bf16" and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return True, torch.bfloat16, "cpu"

    return False, torch.float32, "cpu"

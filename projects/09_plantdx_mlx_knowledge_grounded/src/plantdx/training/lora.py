"""Adapter-method capability check against the installed mlx-vlm backend.

The config schema accepts lora / qlora / dora for forward-compatibility, but the
backend actually installed here (mlx-vlm 0.6.3) implements only LoRA-style
adapters. This module is the single place that says what is runnable, so the
command builder and the runner fail closed on an unsupported method rather than
silently training something other than what was requested.
"""

from __future__ import annotations

from plantdx.core.exceptions import ConfigError
from plantdx.training.config import LoRAConfig

# Methods the mlx-vlm LoRA trainer can actually execute. QLoRA == LoRA over an
# already-quantized (4-bit) base, which is exactly this repo's model.
_BACKEND_SUPPORTED = frozenset({"lora", "qlora"})


def check_method_supported(lora: LoRAConfig) -> None:
    """Raise :class:`ConfigError` if ``lora.method`` is not runnable here."""
    if lora.method not in _BACKEND_SUPPORTED:
        raise ConfigError(
            f"adapter method {lora.method!r} is not supported by the installed "
            f"mlx-vlm backend (0.6.x). Supported: {sorted(_BACKEND_SUPPORTED)}. "
            "Use 'qlora' (recommended for the 4-bit base) or 'lora'."
        )


def is_supported(method: str) -> bool:
    """Return True if the backend can run ``method``."""
    return method in _BACKEND_SUPPORTED

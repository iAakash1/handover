"""
Best-effort peak-memory measurement and RNG seeding across MLX API
versions.

MLX has renamed its memory-tracking functions before
(mx.metal.get_peak_memory() -> mx.get_peak_memory(), etc). These helpers
try the modern name first and fall back to the older one, so the
benchmark keeps running even if your installed mlx version differs. If
neither known API is available, peak memory is reported as None rather
than a fabricated number.
"""

from typing import Optional

import mlx.core as mx


def reset_peak_memory() -> None:
    """Reset MLX's peak-memory counter so the next reading reflects only
    what happens between this call and the following get_peak_memory_mb()."""
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()
    # else: silently no-op — get_peak_memory_mb() will return None too.


def get_peak_memory_mb() -> Optional[float]:
    """Return peak MLX memory usage in MB since the last reset, or None if
    this MLX version exposes neither known peak-memory API."""
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory() / (1024**2)
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory() / (1024**2)
    return None


def set_seed(seed: int) -> bool:
    """
    Seed MLX's RNG for reproducibility. This is a defensive extra layer
    on top of TEMPERATURE=0.0 (greedy decoding, already deterministic on
    its own for most models) in case any model's code path samples
    instead of taking the argmax. Returns True if a seed API was found
    and called, False otherwise — never raises, since a benchmark run
    shouldn't fail over an RNG seeding call.
    """
    if hasattr(mx, "random") and hasattr(mx.random, "seed"):
        mx.random.seed(seed)
        return True
    return False

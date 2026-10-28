"""Seed management for reproducible training.

``resolve_env`` returns the seed-derived environment variables the training
subprocess should inherit (Python hash seed, tokenizer determinism). ``apply``
seeds the RNG libraries in-process, importing mlx/numpy lazily so this module is
import-safe where MLX is not installed (e.g. CI).
"""

from __future__ import annotations

from plantdx.utils.hashing import sha256_hex


def derive_seed(base_seed: int, *tags: str) -> int:
    """Deterministically derive a 32-bit sub-seed from a base seed + tags."""
    return int(sha256_hex(str(base_seed), *tags)[:8], 16)


def resolve_env(seed: int) -> dict[str, str]:
    """Env vars that make the training subprocess reproducible."""
    return {
        "PYTHONHASHSEED": str(seed % (2**32)),
        "TOKENIZERS_PARALLELISM": "false",
    }


def apply(seed: int) -> None:
    """Seed random / numpy / mlx in-process (lazy imports; best-effort)."""
    import random

    random.seed(seed)
    try:  # numpy is usually present; mlx only on Apple Silicon
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except ImportError:
        pass

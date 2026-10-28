"""Deterministic, seed-controlled choice for the corpus.

The corpus is enumerated deterministically, but where one option must be picked
among several (which realization of a concept to use), the choice is a pure
function of a SHA-256 digest over stable string keys — no wall-clock, no RNG
state. Same keys always select the same option, so the corpus is byte-identical
across runs and machines.
"""

from __future__ import annotations

from plantdx.utils.hashing import sha256_hex


def choice_index(n: int, *keys: str) -> int:
    """A deterministic index in ``[0, n)`` derived from ``keys`` (n must be > 0)."""
    if n <= 0:
        raise ValueError("choice_index requires n > 0")
    return int(sha256_hex(*keys), 16) % n

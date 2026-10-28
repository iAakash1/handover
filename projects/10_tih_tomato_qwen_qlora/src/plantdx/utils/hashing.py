"""Hashing helpers.

Centralizes the SHA-256 based hashing used for file checksums, dataset
checksums, and config hashes so the whole pipeline shares one convention
(``caption_framework/00_methodology_overview.md`` §6). Pure functions, CPU-only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_hex(*parts: str | bytes | int) -> str:
    """Return the SHA-256 hex digest of the concatenated parts.

    Parts are separated by a NUL byte so that ``("a", "bc")`` and ``("ab", "c")``
    hash differently. ``int``/``str`` parts are UTF-8 encoded.
    """
    h = hashlib.sha256()
    for i, part in enumerate(parts):
        if i:
            h.update(b"\x00")
        h.update(part if isinstance(part, bytes) else str(part).encode("utf-8"))
    return h.hexdigest()


def stable_json_hash(obj: Mapping[str, Any] | list[Any]) -> str:
    """Return a stable SHA-256 of a JSON-serializable object (sorted keys)."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

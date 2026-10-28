"""Canonical caption-record (de)serialization (doc 04 §1).

The in-memory record type is :class:`plantdx.core.types.CaptionRecord`. This
module owns its JSON(L) serialization, ``schema_version`` handling, and the
round-trip guarantee used by the reproducibility check (doc 00 §6).

(Renamed from ``dataset/schema.py`` to avoid ambiguity with the configuration
schema in ``config/schema.py``.)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plantdx.core.types import CaptionRecord

SCHEMA_VERSION = "1.0"
"""Canonical caption-record schema version (doc 04 §1)."""


def record_to_dict(record: CaptionRecord) -> dict[str, Any]:
    """Serialize a record to a plain JSON-compatible dict."""
    raise NotImplementedError("Milestone 4: record serialization")


def record_from_dict(data: Mapping[str, Any]) -> CaptionRecord:
    """Deserialize a record from a plain dict (inverse of :func:`record_to_dict`)."""
    raise NotImplementedError("Milestone 4: record deserialization")

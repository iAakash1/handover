"""Deterministic template-library statistics (``statistics.json``)."""

from __future__ import annotations

from typing import Any

from plantdx.templates.checksum import content_hash
from plantdx.templates.models import TemplateLibrary


def compute(library: TemplateLibrary, validation_status: str) -> dict[str, Any]:
    """Return the statistics document (all values deterministic)."""
    by_family: dict[str, int] = {}
    by_register: dict[str, int] = {}
    hedged = 0
    for t in library.templates:
        by_family[t.family] = by_family.get(t.family, 0) + 1
        by_register[t.register] = by_register.get(t.register, 0) + 1
        hedged += int(t.hedged)
    return {
        "template_set_version": library.template_set_version,
        "content_hash": content_hash(library),
        "validation_status": validation_status,
        "template_count": len(library.templates),
        "family_count": len(by_family),
        "hedged_count": hedged,
        "by_family": {k: by_family[k] for k in sorted(by_family)},
        "by_register": {k: by_register[k] for k in sorted(by_register)},
    }

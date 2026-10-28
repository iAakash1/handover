"""Deterministic loader for the authored template library (``templates.json``).

Parses the authored asset into typed :class:`Template` objects and cross-checks
its structural integrity. Loading is pure and order-independent; the returned
library is sorted by template id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plantdx.templates.models import (
    SEG_LIST,
    SEG_LIT,
    SEG_OPT,
    SEG_SLOT,
    Segment,
    Template,
    TemplateLibrary,
)
from plantdx.utils.io import read_json

_VERSION = "T1"


def load_library(path: str | Path) -> TemplateLibrary:
    """Load and structurally parse ``templates.json`` into a :class:`TemplateLibrary`."""
    data = read_json(path)
    templates = tuple(
        sorted((_parse_template(t) for t in data.get("templates", [])), key=lambda t: t.id)
    )
    return TemplateLibrary(
        schema_version=str(data.get("schema_version", "1.0.0")),
        template_set_version=str(data.get("template_set_version", _VERSION)),
        families=tuple(data.get("families", [])),
        templates=templates,
        provenance={"source": str(path), "loader": "plantdx.templates"},
    )


def _parse_template(raw: dict[str, Any]) -> Template:
    segments = tuple(_parse_segment(s) for s in raw.get("segments", []))
    return Template(
        id=str(raw["id"]),
        family=str(raw["family"]),
        register=str(raw["register"]),
        length_band=str(raw["length_band"]),
        hedged=bool(raw.get("hedged", False)),
        sign_type_allow=tuple(raw.get("sign_type_allow", [])),
        required=tuple(raw.get("required", [])),
        optional=tuple(raw.get("optional", [])),
        segments=segments,
    )


def _parse_segment(raw: dict[str, Any]) -> Segment:
    if SEG_LIT in raw:
        return Segment(kind=SEG_LIT, text=str(raw[SEG_LIT]))
    if SEG_SLOT in raw:
        return Segment(kind=SEG_SLOT, concept=str(raw[SEG_SLOT]))
    if SEG_OPT in raw:
        return Segment(
            kind=SEG_OPT,
            concept=str(raw[SEG_OPT]),
            glue=str(raw.get("glue", "")),
            suffix=str(raw.get("suffix", "")),
        )
    if SEG_LIST in raw:
        return Segment(
            kind=SEG_LIST,
            concepts=tuple(raw[SEG_LIST]),
            glue=str(raw.get("glue", "")),
            conj=str(raw.get("conj", "and")),
        )
    raise ValueError(f"unknown template segment: {raw!r}")

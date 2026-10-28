"""Canonical serialization of the template library + derived index.

Reuses :func:`plantdx.ontology.domain.serialization.canonical_json`. The
``template_index.json`` artifact is a derived, graph-centric view (by family, by
register, by sign type, by required-concept signature) used by the Sentence
Planner to pick compatible templates without rescanning the library.
"""

from __future__ import annotations

from typing import Any

from plantdx.ontology.domain.serialization import canonical_json
from plantdx.templates.models import (
    SEG_LIT,
    SEG_OPT,
    SEG_SLOT,
    Segment,
    Template,
    TemplateLibrary,
)

__all__ = [
    "canonical_json",
    "index_document",
    "segment_dict",
    "semantic_content",
    "template_dict",
]


def segment_dict(seg: Segment) -> dict[str, Any]:
    """One segment as a canonical dict (only the keys its kind uses)."""
    if seg.kind == SEG_LIT:
        return {"lit": seg.text}
    if seg.kind == SEG_SLOT:
        return {"slot": seg.concept}
    if seg.kind == SEG_OPT:
        return {"opt": seg.concept, "glue": seg.glue, "suffix": seg.suffix}
    return {"list": list(seg.concepts), "glue": seg.glue, "conj": seg.conj}


def template_dict(t: Template) -> dict[str, Any]:
    """One template as a canonical dict."""
    return {
        "id": t.id,
        "family": t.family,
        "register": t.register,
        "length_band": t.length_band,
        "hedged": t.hedged,
        "sign_type_allow": list(t.sign_type_allow),
        "required": list(t.required),
        "optional": list(t.optional),
        "segments": [segment_dict(s) for s in t.segments],
    }


def semantic_content(library: TemplateLibrary) -> dict[str, Any]:
    """The full semantic content excluding provenance (basis of the checksum)."""
    return {
        "schema_version": library.schema_version,
        "template_set_version": library.template_set_version,
        "families": list(library.families),
        "templates": [template_dict(t) for t in sorted(library.templates, key=lambda t: t.id)],
    }


def index_document(library: TemplateLibrary) -> dict[str, Any]:
    """The derived ``template_index.json`` lookup document."""
    by_family: dict[str, list[str]] = {}
    by_register: dict[str, list[str]] = {}
    by_sign_type: dict[str, list[str]] = {}
    for t in library.templates:
        by_family.setdefault(t.family, []).append(t.id)
        by_register.setdefault(t.register, []).append(t.id)
        for st in t.sign_type_allow:
            by_sign_type.setdefault(st, []).append(t.id)
    for index in (by_family, by_register, by_sign_type):
        for ids in index.values():
            ids.sort()
    return {
        "kind": "plantdx.templates.index",
        "schema_version": library.schema_version,
        "template_set_version": library.template_set_version,
        "template_count": len(library.templates),
        "by_family": {k: by_family[k] for k in sorted(by_family)},
        "by_register": {k: by_register[k] for k in sorted(by_register)},
        "by_sign_type": {k: by_sign_type[k] for k in sorted(by_sign_type)},
    }

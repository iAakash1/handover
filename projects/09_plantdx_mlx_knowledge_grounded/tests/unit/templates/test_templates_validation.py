"""Fail-closed tests: the template validator detects structural faults."""

from __future__ import annotations

import dataclasses

import pytest

from plantdx.templates import validator
from plantdx.templates.models import SEG_LIT, SEG_SLOT, Segment, Template, TemplateLibrary


def _template(**over: object) -> Template:
    base = {
        "id": "T-X-01",
        "family": "short",
        "register": "visual",
        "length_band": "short",
        "hedged": False,
        "sign_type_allow": ("lesion",),
        "required": ("primary_sign",),
        "optional": (),
        "segments": (
            Segment(kind=SEG_SLOT, concept="primary_sign"),
            Segment(kind=SEG_LIT, text="."),
        ),
    }
    base.update(over)
    return Template(**base)  # type: ignore[arg-type]


def _library(*templates: Template) -> TemplateLibrary:
    return TemplateLibrary(
        schema_version="1.0.0",
        template_set_version="T1",
        families=(
            "short",
            "single_sentence",
            "two_sentence",
            "clinical",
            "descriptive",
            "educational",
            "dense",
            "long",
        ),
        templates=templates,
    )


@pytest.mark.unit
def test_valid_template_passes() -> None:
    assert validator.collect_violations(_library(_template())) == []


@pytest.mark.unit
def test_duplicate_ids_detected() -> None:
    lib = _library(_template(), _template())
    assert any(v.startswith("V-TPL-1") for v in validator.collect_violations(lib))


@pytest.mark.unit
def test_unknown_family_detected() -> None:
    lib = _library(_template(family="bogus"))
    assert any(v.startswith("V-TPL-2") for v in validator.collect_violations(lib))


@pytest.mark.unit
def test_unknown_concept_id_detected() -> None:
    seg = (Segment(kind=SEG_SLOT, concept="not_a_concept"), Segment(kind=SEG_LIT, text="."))
    lib = _library(_template(required=("not_a_concept",), segments=seg))
    assert any(v.startswith("V-TPL-3") for v in validator.collect_violations(lib))


@pytest.mark.unit
def test_required_mismatch_detected() -> None:
    # Declared required does not match the actual required-slot concepts.
    lib = _library(_template(required=("host",)))
    assert any(v.startswith("V-TPL-4") for v in validator.collect_violations(lib))


@pytest.mark.unit
def test_empty_segments_detected() -> None:
    lib = _library(_template(required=(), segments=()))
    assert any(v.startswith("V-TPL-5") for v in validator.collect_violations(lib))


@pytest.mark.unit
def test_secondary_sign_requires_hedged() -> None:
    seg = (
        Segment(kind=SEG_SLOT, concept="primary_sign"),
        Segment(kind="opt", concept="secondary_sign", glue=" and "),
        Segment(kind=SEG_LIT, text="."),
    )
    lib = _library(_template(optional=("secondary_sign",), segments=seg, hedged=False))
    assert any(v.startswith("V-TPL-6") for v in validator.collect_violations(lib))


@pytest.mark.unit
def test_healthy_routing_detected() -> None:
    lib = _library(_template(sign_type_allow=("healthy",)))  # requires primary_sign -> illegal
    assert any(v.startswith("V-TPL-7") for v in validator.collect_violations(lib))


@pytest.mark.unit
def test_undeclared_family_detected() -> None:
    lib = TemplateLibrary(
        schema_version="1.0.0",
        template_set_version="T1",
        families=("short",),
        templates=(_template(family="clinical"),),
    )
    assert any(v.startswith("V-TPL-8") for v in validator.collect_violations(lib))


@pytest.mark.unit
def test_replace_roundtrip() -> None:
    t = _template()
    assert dataclasses.replace(t, id="T-Y-02").id == "T-Y-02"

"""The template-library validator battery. Fail closed.

Structural, disease-independent checks on the authored ``templates.json``: valid
enums, slot/concept legality, required-slot coherence, hedging discipline, and
healthy-vs-disease routing. Any violation aborts the build. See
``docs/CORPUS.md`` for what each ``V-TPL-*`` rule guards against.
"""

from __future__ import annotations

from plantdx.core.enums import ConceptId, LengthBand, Register, SignType, Style
from plantdx.core.exceptions import PlantDxError
from plantdx.templates.models import SEG_LIST, SEG_LIT, SEG_OPT, SEG_SLOT, Template, TemplateLibrary

CHECK_COUNT = 8

_STYLES = frozenset(s.value for s in Style)
_REGISTERS = frozenset(r.value for r in Register)
_BANDS = frozenset(b.value for b in LengthBand)
_SIGN_TYPES = frozenset(s.value for s in SignType) | {"healthy"}
_CONCEPTS = frozenset(c.value for c in ConceptId)
_SEGMENT_KINDS = frozenset({SEG_LIT, SEG_SLOT, SEG_OPT, SEG_LIST})


class TemplateValidationError(PlantDxError):
    """Raised when the template library violates one or more structural rules."""

    def __init__(self, violations: list[str]) -> None:
        """Initialize the error with the sorted list of rule violations."""
        self.violations = violations
        super().__init__(
            f"template validation failed ({len(violations)} error(s)):\n  "
            + "\n  ".join(violations)
        )


def collect_violations(library: TemplateLibrary) -> list[str]:
    """Run the full battery and return the sorted, deduplicated violation list."""
    violations: list[str] = []
    violations += _v1_unique_ids(library)
    for template in library.templates:
        violations += _v2_enums(template)
        violations += _v3_concept_ids(template)
        violations += _v4_required_coherence(template)
        violations += _v5_segments(template)
        violations += _v6_hedging(template)
        violations += _v7_healthy_routing(template)
        violations += _v8_family_declared(template, library)
    return sorted(set(violations))


def validate(library: TemplateLibrary) -> None:
    """Run the full battery; raise :class:`TemplateValidationError` on any violation."""
    violations = collect_violations(library)
    if violations:
        raise TemplateValidationError(violations)


def _v1_unique_ids(library: TemplateLibrary) -> list[str]:
    v: list[str] = []
    seen: set[str] = set()
    for template in library.templates:
        if template.id in seen:
            v.append(f"V-TPL-1: duplicate template id {template.id!r}")
        seen.add(template.id)
    return v


def _v2_enums(t: Template) -> list[str]:
    v: list[str] = []
    if t.family not in _STYLES:
        v.append(f"V-TPL-2: {t.id} unknown family {t.family!r}")
    if t.register not in _REGISTERS:
        v.append(f"V-TPL-2: {t.id} unknown register {t.register!r}")
    if t.length_band not in _BANDS:
        v.append(f"V-TPL-2: {t.id} unknown length_band {t.length_band!r}")
    for st in t.sign_type_allow:
        if st not in _SIGN_TYPES:
            v.append(f"V-TPL-2: {t.id} unknown sign type {st!r}")
    if not t.sign_type_allow:
        v.append(f"V-TPL-2: {t.id} has empty sign_type_allow")
    return v


def _v3_concept_ids(t: Template) -> list[str]:
    v: list[str] = []
    for cid in _slot_concepts(t) | set(t.required) | set(t.optional):
        if cid not in _CONCEPTS:
            v.append(f"V-TPL-3: {t.id} references unknown concept id {cid!r}")
    return v


def _v4_required_coherence(t: Template) -> list[str]:
    """Declared required/optional must match the concepts the segments actually use."""
    v: list[str] = []
    required_slots = {s.concept for s in t.segments if s.kind == SEG_SLOT}
    if set(t.required) != required_slots:
        v.append(
            f"V-TPL-4: {t.id} declared required {sorted(t.required)} "
            f"!= required-slot concepts {sorted(required_slots)}"
        )
    optional_concepts = {s.concept for s in t.segments if s.kind == SEG_OPT} | {
        c for s in t.segments if s.kind == SEG_LIST for c in s.concepts
    }
    missing = optional_concepts - set(t.optional)
    if missing:
        v.append(f"V-TPL-4: {t.id} uses optional concepts not declared: {sorted(missing)}")
    return v


def _v5_segments(t: Template) -> list[str]:
    v: list[str] = []
    if not t.segments:
        v.append(f"V-TPL-5: {t.id} has no segments")
    for seg in t.segments:
        if seg.kind not in _SEGMENT_KINDS:
            v.append(f"V-TPL-5: {t.id} bad segment kind {seg.kind!r}")
        if seg.kind == SEG_LIT and not seg.text:
            v.append(f"V-TPL-5: {t.id} empty literal segment")
        if seg.kind in (SEG_SLOT, SEG_OPT) and not seg.concept:
            v.append(f"V-TPL-5: {t.id} {seg.kind} segment with no concept")
        if seg.kind == SEG_LIST and not seg.concepts:
            v.append(f"V-TPL-5: {t.id} list segment with no concepts")
    return v


def _v6_hedging(t: Template) -> list[str]:
    """secondary_sign may appear only in a hedged template that carries a hedge word."""
    v: list[str] = []
    uses_secondary = "secondary_sign" in _all_concepts(t)
    if uses_secondary and not t.hedged:
        v.append(f"V-TPL-6: {t.id} uses secondary_sign but is not marked hedged")
    if t.hedged:
        glue = " ".join(s.glue for s in t.segments if s.kind in (SEG_OPT, SEG_LIST))
        lit = " ".join(s.text for s in t.segments if s.kind == SEG_LIT)
        if not any(word in (glue + " " + lit).lower() for word in ("may", "can", "often")):
            v.append(f"V-TPL-6: {t.id} is hedged but carries no hedge word (may/can/often)")
    return v


def _v7_healthy_routing(t: Template) -> list[str]:
    v: list[str] = []
    healthy_only = set(t.sign_type_allow) == {"healthy"}
    concepts = _all_concepts(t)
    if healthy_only:
        if "healthy_state" not in t.required:
            v.append(f"V-TPL-7: {t.id} healthy template must require healthy_state")
        if "primary_sign" in concepts:
            v.append(f"V-TPL-7: {t.id} healthy template must not use primary_sign")
    else:
        if "healthy_state" in concepts:
            v.append(f"V-TPL-7: {t.id} non-healthy template must not use healthy_state")
        if "healthy" in t.sign_type_allow:
            v.append(f"V-TPL-7: {t.id} mixes 'healthy' with disease sign types")
    return v


def _v8_family_declared(t: Template, library: TemplateLibrary) -> list[str]:
    if t.family not in library.families:
        return [f"V-TPL-8: {t.id} family {t.family!r} not in the declared families list"]
    return []


def _slot_concepts(t: Template) -> set[str]:
    return {s.concept for s in t.segments if s.kind in (SEG_SLOT, SEG_OPT)} | {
        c for s in t.segments if s.kind == SEG_LIST for c in s.concepts
    }


def _all_concepts(t: Template) -> set[str]:
    return _slot_concepts(t) | set(t.required) | set(t.optional)

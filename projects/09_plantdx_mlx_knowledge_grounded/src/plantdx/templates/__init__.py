"""Template Engine: load, validate, and index the authored caption templates.

Templates carry *syntax only* (``assets/templates/templates.json``); all domain
content enters through slots naming concept ids. This package loads that asset
into typed objects, validates it structurally (fail closed), derives a lookup
index, and answers the one legality question the Sentence Planner asks: *is this
template compatible with this disease's concept model?*
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plantdx.concepts.models import ConceptModel
from plantdx.templates import checksum, serialization, statistics, validator
from plantdx.templates.loader import load_library
from plantdx.templates.models import Template, TemplateLibrary
from plantdx.templates.validator import TemplateValidationError
from plantdx.utils.io import ensure_dir

__all__ = [
    "ARTIFACT_NAMES",
    "Template",
    "TemplateLibrary",
    "TemplateValidationError",
    "compatible",
    "compatible_templates",
    "compute_statistics",
    "load_library",
    "validate_library",
    "write_artifacts",
]

ARTIFACT_NAMES = (
    "template_index.json",
    "statistics.json",
    "checksum.txt",
    "validation_report.json",
)


def validate_library(library: TemplateLibrary) -> dict[str, Any]:
    """Run the structural validator battery; raise on error, else return the report."""
    violations = validator.collect_violations(library)
    if violations:
        raise TemplateValidationError(violations)
    return {
        "kind": "plantdx.templates.validation_report",
        "status": "valid",
        "checks_run": validator.CHECK_COUNT,
        "violation_count": 0,
        "template_count": len(library.templates),
    }


def compatible(template: Template, model: ConceptModel) -> bool:
    """Whether ``template`` may be used to caption the disease of ``model``.

    A template is compatible iff (a) the disease's effective sign type is allowed
    by the template and (b) every required slot concept is available (mandatory or
    optional) for the disease. This is the sole routing rule (relationship legality).
    """
    if model.sign_type not in template.sign_type_allow:
        return False
    available = set(model.mandatory) | set(model.optional)
    return set(template.required) <= available


def compatible_templates(library: TemplateLibrary, model: ConceptModel) -> list[Template]:
    """All compatible templates for a disease, sorted by id (deterministic)."""
    return [t for t in library.templates if compatible(t, model)]


def compute_statistics(library: TemplateLibrary, validation_status: str) -> dict[str, Any]:
    """Compute the statistics document."""
    return statistics.compute(library, validation_status)


def write_artifacts(
    library: TemplateLibrary,
    out_dir: str | Path,
    stats: dict[str, Any],
    validation_report: dict[str, Any],
) -> list[Path]:
    """Write the four template artifacts deterministically. Returns the written paths."""
    out = ensure_dir(out_dir)
    written: list[Path] = []

    def _write(name: str, text: str) -> None:
        path = out / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    _write(
        "template_index.json",
        serialization.canonical_json(serialization.index_document(library)),
    )
    _write("statistics.json", serialization.canonical_json(stats))
    _write("checksum.txt", checksum.content_hash(library) + "\n")
    _write("validation_report.json", serialization.canonical_json(validation_report))
    return written

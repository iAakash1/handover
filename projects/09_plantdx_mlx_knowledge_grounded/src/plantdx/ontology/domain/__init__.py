"""Domain ontology compiler (``Ontology = f(DKB, Policies)``).

Deterministic: compiling the same DKB with the same policies yields byte-identical
artifacts (no timestamps, no randomness, everything sorted). This package is the
knowledge-graph substrate of ontology_design/; it is separate from the
caption-concept model in ``plantdx.ontology`` (a downstream view / later milestone).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plantdx.ontology.domain import (
    builder,
    checksum,
    serialization,
    statistics,
    validator,
)
from plantdx.ontology.domain.models import Ontology
from plantdx.utils.io import ensure_dir

__all__ = [
    "ARTIFACT_NAMES",
    "CompileResult",
    "Ontology",
    "compile_ontology",
    "compute_statistics",
    "validate_ontology",
    "write_artifacts",
]

ARTIFACT_NAMES = (
    "ontology.json",
    "concept_graph.json",
    "concept_index.json",
    "ontology_statistics.json",
    "ontology_checksum.txt",
    "ontology_build.log",
)


@dataclass
class CompileResult:
    """The outcome of compiling the DKB into the ontology."""

    ontology: Ontology
    dkb: dict[str, Any]
    log: list[str]


def compile_ontology(dkb_path: str | Path) -> CompileResult:
    """Load + validate the DKB, then compile the ontology (no I/O of artifacts)."""
    dkb = builder.load_dkb(dkb_path)
    builder.validate_dkb(dkb)  # DKB validation stage (fail closed)
    dkb_sha = builder.dkb_file_sha256(dkb_path)
    ontology, log = builder.build_ontology(dkb, dkb_sha)
    return CompileResult(ontology=ontology, dkb=dkb, log=log)


def validate_ontology(result: CompileResult) -> None:
    """Run the validator battery on a compiled ontology (raises on any error)."""
    validator.validate(result.ontology, result.dkb)


def compute_statistics(result: CompileResult, validation_status: str) -> dict[str, Any]:
    """Compute the statistics document."""
    return statistics.compute(result.ontology, result.dkb, validation_status)


def write_artifacts(
    result: CompileResult, out_dir: str | Path, stats: dict[str, Any]
) -> list[Path]:
    """Write all six artifacts deterministically. Returns the written paths."""
    out = ensure_dir(out_dir)
    ontology = result.ontology
    written: list[Path] = []

    def _write(name: str, text: str) -> None:
        path = out / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    _write("ontology.json", serialization.canonical_json(serialization.ontology_document(ontology)))
    _write(
        "concept_graph.json",
        serialization.canonical_json(serialization.concept_graph_document(ontology)),
    )
    _write(
        "concept_index.json",
        serialization.canonical_json(serialization.concept_index_document(ontology)),
    )
    _write("ontology_statistics.json", serialization.canonical_json(stats))
    _write("ontology_checksum.txt", checksum.content_hash(ontology) + "\n")
    _write("ontology_build.log", "\n".join(result.log) + "\n")
    return written

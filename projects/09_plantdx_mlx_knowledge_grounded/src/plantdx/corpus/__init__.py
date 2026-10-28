"""Caption corpus layer (planner -> generator -> validator -> builder).

Turns the per-disease Caption Concept Models plus the authored Template Library
into a deterministic, validated, image-independent caption corpus. Every caption
traces to controlled concept realizations (and through them to the DKB and its
evidence); nothing is invented, inferred, or read from an image. Byte-identical
across runs. Exposed as ``plantdx corpus`` / ``plantdx generate`` / ``plantdx
validate`` in the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plantdx.concepts.models import ConceptModelSet
from plantdx.corpus import builder, checksum, serialization, statistics
from plantdx.corpus.builder import CorpusBuildError
from plantdx.corpus.models import Caption, Corpus
from plantdx.templates.models import TemplateLibrary
from plantdx.utils.io import ensure_dir

__all__ = [
    "ARTIFACT_NAMES",
    "Caption",
    "Corpus",
    "CorpusBuildError",
    "build_corpus",
    "compute_statistics",
    "write_artifacts",
]

ARTIFACT_NAMES = (
    "captions.json",
    "captions.jsonl",
    "captions.csv",
    "statistics.json",
    "validation_report.json",
    "checksum.txt",
)


def build_corpus(
    concept_models: ConceptModelSet,
    library: TemplateLibrary,
    *,
    condition: str | None = None,
    crop: str | None = None,
) -> tuple[Corpus, dict[str, Any]]:
    """Build + validate the corpus; return ``(corpus, validation_report)``."""
    corpus, report = builder.build_corpus(concept_models, library, condition=condition, crop=crop)
    corpus.provenance["content_hash"] = checksum.content_hash(corpus)
    return corpus, report


def compute_statistics(corpus: Corpus, validation_status: str) -> dict[str, Any]:
    """Compute the statistics document."""
    return statistics.compute(corpus, validation_status)


def write_artifacts(
    corpus: Corpus,
    out_dir: str | Path,
    stats: dict[str, Any],
    validation_report: dict[str, Any],
) -> list[Path]:
    """Write all six corpus artifacts deterministically. Returns the written paths."""
    out = ensure_dir(out_dir)
    written: list[Path] = []

    def _write(name: str, text: str) -> None:
        path = out / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    _write(
        "captions.json",
        serialization.canonical_json(serialization.captions_document(corpus)),
    )
    _write("captions.jsonl", serialization.jsonl_text(corpus))
    _write("captions.csv", serialization.csv_text(corpus))
    _write("statistics.json", serialization.canonical_json(stats))
    _write("validation_report.json", serialization.canonical_json(validation_report))
    _write("checksum.txt", corpus.provenance["content_hash"] + "\n")
    return written

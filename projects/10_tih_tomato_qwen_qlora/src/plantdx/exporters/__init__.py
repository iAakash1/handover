"""Dataset Exporters: reshape the one caption corpus into training formats.

Every exporter consumes the SAME :class:`~plantdx.corpus.models.Corpus` and
differs only in serialization (``generic``/``llava``/``paligemma``/``blip2``/
``messages``). Deterministic and byte-identical: the record order follows the
corpus order and no exporter alters caption text. Each export writes a
``data.jsonl`` plus a ``manifest.json`` pinning the source corpus content hash
and the export's own checksum. Exposed as ``plantdx corpus --format`` /
``--all`` in the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plantdx.corpus.models import Corpus
from plantdx.exporters.formats import FORMAT_REGISTRY, FORMATS
from plantdx.utils.hashing import sha256_bytes
from plantdx.utils.io import ensure_dir

__all__ = [
    "FORMATS",
    "export_checksum",
    "export_jsonl",
    "export_records",
    "write_all",
    "write_export",
]


def _sorted_captions(corpus: Corpus) -> list[Any]:
    return sorted(corpus.captions, key=lambda c: (c.disease_id, c.template_id, c.caption_id))


def export_records(corpus: Corpus, fmt: str) -> list[dict[str, Any]]:
    """Reshape every caption into ``fmt``'s record shape (deterministic order)."""
    if fmt not in FORMAT_REGISTRY:
        raise KeyError(f"unknown export format {fmt!r}; known: {', '.join(FORMATS)}")
    reshape = FORMAT_REGISTRY[fmt]
    return [reshape(c) for c in _sorted_captions(corpus)]


def export_jsonl(corpus: Corpus, fmt: str) -> str:
    """One JSON record per line for ``fmt`` (sorted keys, deterministic)."""
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in export_records(corpus, fmt)]
    return "\n".join(lines) + ("\n" if lines else "")


def export_checksum(corpus: Corpus, fmt: str) -> str:
    """``sha256:<hex>`` of the exported JSONL bytes."""
    return "sha256:" + sha256_bytes(export_jsonl(corpus, fmt).encode("utf-8"))


def write_export(corpus: Corpus, fmt: str, out_dir: str | Path) -> list[Path]:
    """Write ``<out_dir>/<fmt>/data.jsonl`` + ``manifest.json``. Returns written paths."""
    out = ensure_dir(Path(out_dir) / fmt)
    jsonl = export_jsonl(corpus, fmt)
    manifest = {
        "kind": "plantdx.exporters.manifest",
        "format": fmt,
        "record_count": len(corpus.captions),
        "corpus_content_hash": corpus.provenance.get("content_hash", ""),
        "export_checksum": export_checksum(corpus, fmt),
    }
    data_path = out / "data.jsonl"
    manifest_path = out / "manifest.json"
    data_path.write_text(jsonl, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return [data_path, manifest_path]


def write_all(corpus: Corpus, out_root: str | Path) -> list[Path]:
    """Write every registered format under ``out_root/<fmt>/``."""
    written: list[Path] = []
    for fmt in FORMATS:
        written.extend(write_export(corpus, fmt, out_root))
    return written

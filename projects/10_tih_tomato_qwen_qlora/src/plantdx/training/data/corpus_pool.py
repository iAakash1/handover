"""Load the frozen caption corpus as a per-disease response pool (read-only).

The corpus is the RESPONSE POOL for training: captions are used verbatim as the
target text. This module never mutates or re-derives it — it only reads
``artifacts/corpus/captions.jsonl`` and groups the relevant captions by disease.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from plantdx.core.exceptions import DerivationError
from plantdx.utils.io import read_jsonl


@dataclass(frozen=True)
class CaptionPool:
    """Captions grouped by ``disease_id`` (each list sorted by caption_id)."""

    by_disease: dict[str, tuple[str, ...]]  # disease_id -> (caption text, ...)
    source_checksum: str  # corpus content_hash pinned into the manifest

    def texts(self, disease_id: str) -> tuple[str, ...]:
        """Return the caption texts for ``disease_id`` (empty tuple if none)."""
        return self.by_disease.get(disease_id, ())


def load_caption_pool(corpus_path: str | Path, disease_ids: set[str]) -> CaptionPool:
    """Read the frozen corpus and keep only captions for ``disease_ids``.

    Captions are keyed by their ``condition`` (the DKB disease id) and sorted by
    ``caption_id`` for determinism. Raises :class:`DerivationError` if the corpus
    is missing (run ``plantdx generate`` first) or a requested disease has none.
    """
    path = Path(corpus_path)
    if not path.is_file():
        raise DerivationError(
            f"caption corpus not found at {path}. Build the (frozen) corpus first:\n"
            f"    plantdx generate"
        )

    grouped: dict[str, list[tuple[str, str]]] = {d: [] for d in disease_ids}
    for row in read_jsonl(path):
        disease = str(row.get("condition", ""))
        if disease not in grouped:
            continue
        text = str(row.get("text", "")).strip()
        cap_id = str(row.get("caption_id", ""))
        if text:
            grouped[disease].append((cap_id, text))

    # The corpus's own content hash lives in the sibling checksum.txt.
    checksum_file = path.with_name("checksum.txt")
    source_checksum = checksum_file.read_text().strip() if checksum_file.is_file() else ""

    by_disease: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    for disease, pairs in grouped.items():
        if not pairs:
            missing.append(disease)
            continue
        pairs.sort(key=lambda pc: pc[0])  # by caption_id
        by_disease[disease] = tuple(text for _cid, text in pairs)
    if missing:
        raise DerivationError("the corpus has no captions for: " + ", ".join(sorted(missing)))
    return CaptionPool(by_disease=by_disease, source_checksum=source_checksum)

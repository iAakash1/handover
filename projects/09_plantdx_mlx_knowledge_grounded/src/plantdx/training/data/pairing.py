"""Pair each image with an instruction and caption(s) -> training rows.

Deterministic: every choice (which caption, which instruction) is a pure
function of a SHA-256 seed fanout from ``(split_seed, image_id, k)``. The caption
text is taken verbatim from the frozen corpus pool; nothing is generated here.
"""

from __future__ import annotations

from dataclasses import dataclass

from plantdx.training.data.corpus_pool import CaptionPool
from plantdx.training.data.discovery import ImageItem
from plantdx.utils.hashing import sha256_hex


@dataclass(frozen=True)
class TrainingRow:
    """One (image, instruction, response) example, plus provenance."""

    image_id: str
    image_path: str
    disease_id: str
    class_name: str
    instruction: str
    response: str


def _pick(seed_parts: list[str], n: int) -> int:
    """Deterministic index in [0, n) from a SHA-256 of the seed parts."""
    return int(sha256_hex(*seed_parts)[:8], 16) % n


def build_rows(
    images: list[ImageItem],
    pool: CaptionPool,
    instructions: tuple[str, ...],
    *,
    seed: int,
    captions_per_image: int,
    max_captions_per_disease: int,
) -> list[TrainingRow]:
    """Build ``captions_per_image`` rows per image, deterministically.

    For each image, its disease caption pool is truncated to
    ``max_captions_per_disease`` (already sorted by caption_id), then
    ``captions_per_image`` distinct captions are chosen by seeded stride; the
    instruction is chosen independently per row. Rows are returned in image order.
    """
    rows: list[TrainingRow] = []
    for item in images:
        texts = pool.texts(item.disease_id)[:max_captions_per_disease]
        if not texts:
            continue
        k = min(captions_per_image, len(texts))
        start = _pick([str(seed), item.image_id, "caption"], len(texts))
        stride = max(1, len(texts) // k)
        for j in range(k):
            cap_idx = (start + j * stride) % len(texts)
            instr_idx = _pick([str(seed), item.image_id, "instruction", str(j)], len(instructions))
            rows.append(
                TrainingRow(
                    image_id=item.image_id,
                    image_path=item.path,
                    disease_id=item.disease_id,
                    class_name=item.class_name,
                    instruction=instructions[instr_idx],
                    response=texts[cap_idx],
                )
            )
    return rows

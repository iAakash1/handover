"""Assemble and write the mlx-vlm training dataset (deterministic, tomato-only).

Pipeline: label map -> discover images -> load frozen caption pool -> pair ->
image-grouped stratified split -> write ``train/validation/test.jsonl`` +
``manifest.json``. Each JSONL row is what ``mlx_vlm.lora`` consumes:
``{"image": <abs path>, "question": <instruction>, "answer": <caption>}`` for the
``qa`` format, or a ``messages`` chat structure for the ``messages`` format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plantdx.core.exceptions import DerivationError
from plantdx.training.config import DataConfig
from plantdx.training.data.corpus_pool import load_caption_pool
from plantdx.training.data.discovery import discover_images
from plantdx.training.data.instructions import load_instructions
from plantdx.training.data.label_map import load_label_map
from plantdx.training.data.pairing import TrainingRow, build_rows
from plantdx.training.data.splits import assign_splits
from plantdx.utils.io import ensure_dir, write_json, write_jsonl

_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetStats:
    """Summary of a built dataset (for the plan/report — no side effects)."""

    output_dir: str
    image_count: int
    row_count: int
    per_split: dict[str, int]
    per_disease: dict[str, int]
    corpus_checksum: str
    instruction_count: int


def _row_payload(row: TrainingRow, response_format: str) -> dict[str, Any]:
    # mlx-vlm inserts the image token itself from the `image` column + num_images;
    # the text must NOT contain a literal <image> marker (it would double-insert).
    if response_format == "messages":
        return {
            "image": row.image_path,
            "messages": [
                {"role": "user", "content": row.instruction},
                {"role": "assistant", "content": row.response},
            ],
        }
    return {"image": row.image_path, "question": row.instruction, "answer": row.response}


def build_training_dataset(
    data: DataConfig,
    *,
    output_dir: str | Path,
    label_map_asset: str | Path | None = None,
    instructions_asset: str | Path | None = None,
) -> DatasetStats:
    """Build the dataset and write it under ``output_dir``; return its stats."""
    label_map = (
        load_label_map(data.crop, asset_path=label_map_asset)
        if label_map_asset
        else load_label_map(data.crop)
    )
    # Honor the configured class subset (tomato-only, exactly the 10 classes).
    allowed = set(data.classes)
    label_map = {c: d for c, d in label_map.items() if c in allowed}
    if set(label_map) != allowed:
        missing = sorted(allowed - set(label_map))
        raise DerivationError(f"label map is missing configured classes: {missing}")

    images = discover_images(data.processed_dir, data.crop, label_map, image_glob=data.image_glob)
    disease_ids = {item.disease_id for item in images}
    pool = load_caption_pool(data.corpus_path, disease_ids)
    instructions = (
        load_instructions(asset_path=instructions_asset)
        if instructions_asset
        else load_instructions()
    )

    rows = build_rows(
        images,
        pool,
        instructions,
        seed=data.split_seed,
        captions_per_image=data.captions_per_image,
        max_captions_per_disease=data.max_captions_per_disease,
    )
    split_of = assign_splits(
        images,
        seed=data.split_seed,
        train_ratio=data.train_ratio,
        val_ratio=data.val_ratio,
        test_ratio=data.test_ratio,
    )

    out = ensure_dir(output_dir)
    per_split: dict[str, int] = dict.fromkeys(_SPLITS, 0)
    per_disease: dict[str, int] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {s: [] for s in _SPLITS}
    for row in rows:
        split = split_of[row.image_id]
        split_rows[split].append(_row_payload(row, data.response_format))
        per_split[split] += 1
        per_disease[row.disease_id] = per_disease.get(row.disease_id, 0) + 1

    for split in _SPLITS:
        write_jsonl(out / f"{split}.jsonl", split_rows[split])

    stats = DatasetStats(
        output_dir=str(out),
        image_count=len(images),
        row_count=len(rows),
        per_split=per_split,
        per_disease=dict(sorted(per_disease.items())),
        corpus_checksum=pool.source_checksum,
        instruction_count=len(instructions),
    )
    _write_manifest(out, data, stats)
    return stats


def _write_manifest(out: Path, data: DataConfig, stats: DatasetStats) -> None:
    write_json(
        out / "manifest.json",
        {
            "crop": data.crop,
            "classes": list(data.classes),
            "response_format": data.response_format,
            "captions_per_image": data.captions_per_image,
            "split_seed": data.split_seed,
            "split_ratios": {
                "train": data.train_ratio,
                "validation": data.val_ratio,
                "test": data.test_ratio,
            },
            "image_count": stats.image_count,
            "row_count": stats.row_count,
            "rows_per_split": stats.per_split,
            "rows_per_disease": stats.per_disease,
            "instruction_count": stats.instruction_count,
            "corpus_checksum": stats.corpus_checksum,
            "files": {s: f"{s}.jsonl" for s in _SPLITS},
        },
    )

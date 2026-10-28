"""
Stage 4 — Train / validation / test split.

WHY THIS EXISTS
    Because the unit of supervision (a caption) is not the unit of independence
    (an image). Five captions describe the same pixels, so splitting at the row
    level would put four descriptions of a leaf in training and the fifth in
    validation — and the resulting validation loss would be meaningless.

    Splitting is therefore done on IMAGES and the captions follow. This is the
    single most important correctness property of the whole data pipeline, which
    is why it is a module of its own rather than four lines inside the builder.

INPUT   pair rows carrying image_id and disease
OUTPUT  {"train": [...], "val": [...], "test": [...]}, each row tagged with split

STRATEGY: grouped_rotation
    Within every class the images are rotated by `fold` and then cut into
    train / val / test in configured sizes. Deterministic, perfectly stratified,
    and running folds 0..N-1 tests every image exactly once — which is the only
    honest way to report a number from a dataset of this size.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List

from ..config import Config


def grouped_rotation(cfg: Config, image_numbers: List[int]) -> Dict[int, str]:
    """Map image number -> split name for one class."""
    n = cfg.data.expected_images_per_class
    ordered = sorted(image_numbers)
    if len(ordered) != n:
        raise ValueError(f"expected {n} images per class, got {len(ordered)}")

    rotated = [ordered[(i + cfg.split.fold) % n] for i in range(n)]
    sizes = [("train", cfg.split.train_per_class),
             ("val", cfg.split.val_per_class),
             ("test", cfg.split.test_per_class)]

    assignment: Dict[int, str] = {}
    cursor = 0
    for name, size in sizes:
        for image_no in rotated[cursor:cursor + size]:
            assignment[image_no] = name
        cursor += size
    return assignment


STRATEGIES: Dict[str, Callable] = {
    "grouped_rotation": grouped_rotation,
}


def split_rows(cfg: Config, rows: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Assign every row to a split via its image, then group.

    The returned rows are copies carrying a "split" field, so a single JSONL of
    all pairs remains a valid, self-describing artifact.
    """
    strategy = STRATEGIES.get(cfg.split.strategy)
    if strategy is None:
        raise ValueError(f"unknown split strategy '{cfg.split.strategy}'. "
                         f"Available: {sorted(STRATEGIES)}")

    per_class_images: Dict[str, set] = defaultdict(set)
    image_number: Dict[str, int] = {}
    for row in rows:
        per_class_images[row["disease"]].add(row["image_id"])
        image_number[row["image_id"]] = int(row["image_id"].rsplit("img", 1)[1])

    image_split: Dict[str, str] = {}
    for disease, image_ids in per_class_images.items():
        numbers = [image_number[i] for i in image_ids]
        assignment = strategy(cfg, numbers)
        for image_id in image_ids:
            image_split[image_id] = assignment[image_number[image_id]]

    splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    for row in rows:
        name = image_split[row["image_id"]]
        splits[name].append({**row, "split": name})
    return splits


def summarize(splits: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Compact description of a split, for logging and the manifest."""
    out: Dict[str, Dict] = {}
    for name, rows in splits.items():
        images = sorted({row["image_id"] for row in rows})
        per_class: Dict[str, int] = defaultdict(int)
        for image_id in images:
            disease = next(r["disease"] for r in rows if r["image_id"] == image_id)
            per_class[disease] += 1
        out[name] = {
            "n_images": len(images),
            "n_rows": len(rows),
            "images": images,
            "images_per_class": dict(sorted(per_class.items())),
        }
    return out

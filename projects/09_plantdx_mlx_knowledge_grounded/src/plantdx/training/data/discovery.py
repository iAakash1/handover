"""Enumerate normalized tomato images (paths + folder labels only, no pixels).

Walks ``<processed_dir>/<crop>/processed/<class>/`` in a fully sorted order so
the result is byte-stable across machines and filesystems.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from plantdx.core.exceptions import DerivationError


@dataclass(frozen=True)
class ImageItem:
    """One discovered image: a stable id, its path, and its resolved labels."""

    image_id: str  # "<class>/<filename>" — stable, path-independent
    path: str  # absolute path as a string (for the JSONL rows)
    class_name: str  # normalized folder name, e.g. "early_blight"
    disease_id: str  # DKB id, e.g. "tomato_early_blight"


def discover_images(
    processed_dir: str | Path,
    crop: str,
    label_map: dict[str, str],
    *,
    image_glob: str = "*.JPG",
) -> list[ImageItem]:
    """Discover every image for the configured ``crop`` classes, deterministically.

    Only folders present in ``label_map`` are read (so a stray folder can never
    leak in). Raises :class:`DerivationError` with an actionable message if the
    processed tree is missing (the ``plantdx normalize`` prerequisite).
    """
    root = Path(processed_dir) / crop / "processed"
    if not root.is_dir():
        raise DerivationError(
            f"normalized images not found at {root}. Run the (frozen) normalizer first:\n"
            f"    plantdx normalize --dataset {crop}"
        )

    items: list[ImageItem] = []
    for class_name in sorted(label_map):
        disease_id = label_map[class_name]
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob(image_glob), key=lambda p: p.name):
            if not path.is_file():
                continue
            items.append(
                ImageItem(
                    image_id=f"{class_name}/{path.name}",
                    path=str(path.resolve()),
                    class_name=class_name,
                    disease_id=disease_id,
                )
            )
    if not items:
        raise DerivationError(
            f"no images matched {image_glob!r} under {root}. Check the glob and that "
            f"`plantdx normalize --dataset {crop}` populated the class folders."
        )
    return items

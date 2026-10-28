"""
Dataset traversal.

Folder names under DATASET_DIR (e.g. "early_blight/") exist only to
organize images on disk and to make console progress and image_metadata
readable/traceable while the benchmark runs. They are never sent to a
model as part of the prompt, and never compared against a model's
prediction anywhere in this codebase. This phase collects raw
predictions only; evaluation against expert ground truth is a separate,
later phase.
"""

from pathlib import Path
from typing import List, Optional, TypedDict

from config import DATASET_DIR, IMAGE_EXTENSIONS
from metadata_utils import ImageMetadata, compute_image_metadata


class DatasetItem(TypedDict):
    image_path: str
    class_folder: str
    image_metadata: ImageMetadata


def discover_images(
    dataset_dir: Path = DATASET_DIR,
    limit_per_class: Optional[int] = None,
) -> List[DatasetItem]:
    """
    Walk dataset_dir/<class_folder>/<image> and return a flat, sorted list
    of DatasetItem dicts, each with the resolved image path, its class
    folder, and full image_metadata (dimensions, file size, SHA-256 hash,
    relative path) computed once here and reused for every model, since
    none of it depends on which model is being run.

    Args:
        dataset_dir: root directory containing one subfolder per class.
        limit_per_class: if set, only the first N images (alphabetically)
            of each class folder are included. Useful for a quick smoke
            test before running the full set.

    class_folder / image_metadata.class_folder are included purely for
    progress logging and traceability back to the original file layout —
    they are not used in inference or in any comparison/evaluation logic.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Place your image folders there, or update DATASET_DIR in scripts/config.py."
        )

    items: List[DatasetItem] = []
    class_folders = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    for class_folder in class_folders:
        images = sorted(
            p for p in class_folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit_per_class is not None:
            images = images[:limit_per_class]
        for image_path in images:
            resolved = image_path.resolve()
            items.append(
                {
                    "image_path": str(resolved),
                    "class_folder": class_folder.name,
                    "image_metadata": compute_image_metadata(
                        resolved, dataset_dir.resolve(), class_folder.name
                    ),
                }
            )

    if not items:
        raise FileNotFoundError(f"No images found under {dataset_dir}")

    return items

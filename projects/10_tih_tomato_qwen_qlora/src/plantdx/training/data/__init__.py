"""Training-data assembly: image x frozen-corpus cross-join -> mlx-vlm JSONL.

CPU-only, deterministic, tomato-only. Reads image *paths* and folder labels
(never pixels) and pairs each image with an instruction and a caption drawn
verbatim from the frozen corpus. No stage here modifies the corpus, ontology,
vocabulary, concepts, templates, or the DKB.
"""

from __future__ import annotations

from plantdx.training.data.builder import DatasetStats, build_training_dataset
from plantdx.training.data.corpus_pool import CaptionPool, load_caption_pool
from plantdx.training.data.discovery import ImageItem, discover_images
from plantdx.training.data.instructions import load_instructions
from plantdx.training.data.label_map import load_label_map
from plantdx.training.data.pairing import TrainingRow, build_rows
from plantdx.training.data.splits import assign_splits

__all__ = [
    "CaptionPool",
    "DatasetStats",
    "ImageItem",
    "TrainingRow",
    "assign_splits",
    "build_rows",
    "build_training_dataset",
    "discover_images",
    "load_caption_pool",
    "load_instructions",
    "load_label_map",
]

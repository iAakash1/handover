"""
Data layer — everything between an annotation artifact and a training-ready JSONL.

    sources.py   where validated captions come from      (swap point)
    pairs.py     bind captions to verified image files
    verify.py    invariant gate; refuses to pass bad data
    splits.py    group by image, stratify by class

The public entry point is build_dataset(), which runs all four in order and is
what scripts/01_build_dataset.py calls.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from ..config import Config
from ..io import read_jsonl, write_json, write_jsonl
from .pairs import Defect, ImageInfo, build_pairs, images_as_dict
from .sources import CaptionRecord, get_source, normalize_whitespace
from .splits import split_rows, summarize
from .verify import CheckReport, verify_dataset

__all__ = [
    "build_dataset", "load_split", "get_source", "build_pairs", "split_rows",
    "verify_dataset", "summarize", "normalize_whitespace",
    "CaptionRecord", "ImageInfo", "Defect", "CheckReport",
]


def build_dataset(cfg: Config, verify_provenance: bool = False,
                  write: bool = True) -> Dict:
    """
    Run the whole data layer and (optionally) persist it.

    Returns a result dict with rows, images, defects, splits, the check report,
    and the manifest — so callers can inspect everything without re-reading disk.
    Raises RuntimeError if verification fails: a bad dataset must never reach
    training.
    """
    source = get_source(cfg)
    records = source.load()

    rows, images, defects = build_pairs(cfg, records, verify_provenance)
    splits = split_rows(cfg, rows)
    report = verify_dataset(cfg, rows, images, splits)

    if not report.ok:
        raise RuntimeError(
            "dataset verification failed:\n" + report.render())

    manifest = {
        "config_file": cfg.source_file,
        "instruction": cfg.data.instruction,
        "source": source.fingerprint(),
        "classes": sorted(cfg.data.source.class_map),
        "n_images": len(images),
        "n_pairs": len(rows),
        "captions_per_image": cfg.data.expected_captions_per_image,
        "split": {"strategy": cfg.split.strategy, "fold": cfg.split.fold,
                  **summarize(splits)},
        "flagged_captions": len(defects),
        "images": images_as_dict(images),
    }

    if write:
        write_jsonl(rows, cfg.paths.pairs)
        for name in ("train", "val", "test"):
            write_jsonl(splits[name], cfg.paths.split_file(name))
        write_json(manifest, cfg.paths.manifest)
        write_json([asdict(d) for d in defects], cfg.paths.defects)

    return {"rows": rows, "images": images, "defects": defects,
            "splits": splits, "report": report, "manifest": manifest,
            "source": source}


def load_split(cfg: Config, split: str) -> List[Dict]:
    """Read a previously built split."""
    return read_jsonl(cfg.paths.split_file(split))

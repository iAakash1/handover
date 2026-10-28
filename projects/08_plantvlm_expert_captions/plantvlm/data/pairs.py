"""
Stage 2 — Image/caption pair creation.

WHY THIS EXISTS
    A caption on its own is not supervision. Supervision is a caption bound to a
    specific image file that is proven to exist, proven to be readable, and
    proven to be the file the annotator was actually looking at. This module
    performs that binding and records the evidence.

INPUT   CaptionRecords from a source + the image tree on disk
OUTPUT  Pair rows (the JSONL contract) + one ImageInfo per image + defect flags

THE JSONL CONTRACT
    Every downstream stage reads these fields and no others:

        id             stable unique key, traceable to a spreadsheet cell
        image          absolute path
        image_id       GROUPING KEY — the five captions of an image share it,
                       which is what makes leak-free splitting possible
        disease        class label (used for splitting and evaluation only)
        caption_index  1..5
        instruction    the question asked at train and inference time
        caption        the validated caption, verbatim
        sha256         proof of which image file this is
        origin         spreadsheet cell the caption came from

DEFECTS ARE REPORTED, NEVER REPAIRED
    A caption that references another image, or that ends mid-sentence, is
    invalid supervision for a single-image model — but it is also validated human
    work. This module flags it and moves on. Deciding what to do about it is the
    annotator's call, not the pipeline's.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ..config import Config
from ..io import sha256_file
from .sources import CaptionRecord


@dataclass
class ImageInfo:
    image_id: str
    disease: str
    image_no: int
    path: str
    sha256: str
    width: int
    height: int
    mode: str
    provenance: Optional[str] = None


@dataclass
class Defect:
    id: str
    origin: str
    disease: str
    image_no: int
    caption_index: int
    flags: List[str]
    caption: str


# ── Defect detection ─────────────────────────────────────────────────────────

def inspect_caption(caption: str, cfg: Config) -> List[str]:
    """Return defect tags for one caption (empty list = clean)."""
    flags: List[str] = []
    if not caption.rstrip().endswith((".", "!", "?")):
        flags.append("TRUNCATED_no_terminal_punctuation")
    for phrase in cfg.eval.cross_image_phrases:
        if re.search(re.escape(phrase), caption, re.IGNORECASE):
            flags.append(f"CROSS_IMAGE_REFERENCE:{phrase}")
    if re.search(r"the image being", caption, re.IGNORECASE):
        flags.append("META_IMAGE_QUALITY")
    if len(caption.split()) < 15:
        flags.append("VERY_SHORT_under_15_words")
    return flags


# ── Image resolution ─────────────────────────────────────────────────────────

def make_image_id(cfg: Config, disease: str, image_no: int) -> str:
    return f"{cfg.data.source.class_map[disease]}__img{image_no}"


def resolve_image(cfg: Config, disease: str, image_no: int,
                  provenance_index: Optional[Dict[str, str]] = None) -> ImageInfo:
    """Locate exactly one image file, open it, hash it, and record what it is."""
    folder = cfg.paths.image_root / cfg.data.source.class_map[disease]
    if not folder.exists():
        raise FileNotFoundError(f"image folder missing: {folder}")

    matches = [p for p in folder.glob(f"{image_no}.*")
               if p.suffix in cfg.data.valid_extensions]
    if not matches:
        raise FileNotFoundError(f"no image for '{disease}' #{image_no} in {folder}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous image for '{disease}' #{image_no}: "
                         f"{[m.name for m in matches]}")

    path = matches[0]
    with Image.open(path) as im:
        width, height, mode = im.width, im.height, im.mode

    digest = sha256_file(path)
    return ImageInfo(
        image_id=make_image_id(cfg, disease, image_no),
        disease=disease,
        image_no=image_no,
        path=path.resolve().as_posix(),
        sha256=digest,
        width=width, height=height, mode=mode,
        provenance=(provenance_index or {}).get(digest),
    )


def build_provenance_index(cfg: Config) -> Dict[str, str]:
    """
    hash -> 'SourceFolder/filename' for the original dataset.

    Optional and slow (it hashes thousands of files), but it is what turns "these
    are tomato leaves" into "these are exactly these files from this public
    dataset" — the difference between a claim and a citation.
    """
    if not cfg.paths.provenance_root or not cfg.paths.provenance_root.exists():
        return {}
    index: Dict[str, str] = {}
    for disease, folder_name in cfg.data.provenance_map.items():
        folder = cfg.paths.provenance_root / folder_name
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.suffix in cfg.data.valid_extensions:
                index[sha256_file(f)] = f"{folder.name}/{f.name}"
    return index


# ── Pair construction ────────────────────────────────────────────────────────

def build_pairs(cfg: Config, records: List[CaptionRecord],
                verify_provenance: bool = False,
                ) -> Tuple[List[Dict], Dict[str, ImageInfo], List[Defect]]:
    """
    Bind every caption to its image.

    Returns (rows, images_by_id, defects). Rows carry no split assignment yet —
    splitting is a separate, swappable stage.
    """
    provenance_index = build_provenance_index(cfg) if verify_provenance else {}

    images: Dict[str, ImageInfo] = {}
    for rec in records:
        image_id = make_image_id(cfg, rec.disease, rec.image_no)
        if image_id not in images:
            images[image_id] = resolve_image(
                cfg, rec.disease, rec.image_no, provenance_index)

    rows: List[Dict] = []
    defects: List[Defect] = []

    for rec in records:
        image_id = make_image_id(cfg, rec.disease, rec.image_no)
        info = images[image_id]
        row_id = f"{image_id}__c{rec.caption_index}"

        flags = inspect_caption(rec.caption, cfg)
        if flags:
            defects.append(Defect(
                id=row_id, origin=rec.origin, disease=rec.disease,
                image_no=rec.image_no, caption_index=rec.caption_index,
                flags=flags, caption=rec.caption))

        rows.append({
            "id": row_id,
            "image": info.path,
            "image_id": image_id,
            "disease": rec.disease,
            "caption_index": rec.caption_index,
            "instruction": cfg.data.instruction,
            "caption": rec.caption,
            "n_words": len(rec.caption.split()),
            "sha256": info.sha256,
            "origin": rec.origin,
        })

    return rows, images, defects


def images_as_dict(images: Dict[str, ImageInfo]) -> Dict[str, Dict]:
    return {k: asdict(v) for k, v in images.items()}

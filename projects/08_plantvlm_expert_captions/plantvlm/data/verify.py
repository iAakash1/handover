"""
Stage 3 — Dataset verification. A GATE, not a report.

WHY THIS EXISTS
    Every expensive failure in a fine-tuning project is a data failure that was
    discovered late: a class with four images instead of five, a caption bound to
    the wrong file, an image that appears in both train and validation. Each of
    these produces a loss curve that looks completely normal.

    This module states the invariants explicitly and refuses to let the pipeline
    continue when one is broken. It runs in under a second and it is the cheapest
    insurance in the project.

INPUT   pair rows + image records (+ optionally the finished splits)
OUTPUT  a CheckReport; callers abort when ok is False

WHY IT IS A SEPARATE MODULE
    Verification that lives inside the builder only runs when you build. Here it
    is callable on its own (`python scripts/00_verify.py`), so the dataset can be
    re-checked at any time — after an annotator edits the workbook, after a file
    is moved, before a paper deadline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import Config
from .pairs import ImageInfo


@dataclass
class CheckReport:
    passed: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        (self.passed if condition else self.failures).append(
            label if condition or not detail else f"{label}: {detail}")

    def warn(self, condition: bool, label: str) -> None:
        if not condition:
            self.warnings.append(label)

    def render(self) -> str:
        lines = []
        for item in self.passed:
            lines.append(f"  PASS  {item}")
        for item in self.warnings:
            lines.append(f"  WARN  {item}")
        for item in self.failures:
            lines.append(f"  FAIL  {item}")
        lines.append("")
        lines.append(f"  {len(self.passed)} passed, {len(self.warnings)} warnings, "
                     f"{len(self.failures)} failures")
        return "\n".join(lines)


def verify_dataset(cfg: Config,
                   rows: List[Dict],
                   images: Dict[str, ImageInfo],
                   splits: Optional[Dict[str, List[Dict]]] = None) -> CheckReport:
    """Run every invariant. Structural problems are failures; quality issues warn."""
    r = CheckReport()
    n_classes = len(cfg.data.source.class_map)

    # ── Coverage ─────────────────────────────────────────────────────────────
    by_class_images = defaultdict(set)
    for info in images.values():
        by_class_images[info.disease].add(info.image_no)

    r.check(len(by_class_images) == n_classes,
            f"all {n_classes} configured classes present",
            f"found {sorted(by_class_images)}")

    bad = {d: sorted(v) for d, v in by_class_images.items()
           if len(v) != cfg.data.expected_images_per_class}
    r.check(not bad,
            f"every class has {cfg.data.expected_images_per_class} images",
            str(bad))

    per_image = defaultdict(int)
    for row in rows:
        per_image[row["image_id"]] += 1
    wrong = {k: v for k, v in per_image.items()
             if v != cfg.data.expected_captions_per_image}
    r.check(not wrong,
            f"every image has {cfg.data.expected_captions_per_image} captions",
            str(wrong))

    expected_rows = (n_classes * cfg.data.expected_images_per_class
                     * cfg.data.expected_captions_per_image)
    r.check(len(rows) == expected_rows,
            f"pair count is {expected_rows}", f"got {len(rows)}")

    # ── Integrity ────────────────────────────────────────────────────────────
    r.check(len({row["id"] for row in rows}) == len(rows),
            "row ids are unique")
    r.check(all(row["caption"].strip() for row in rows),
            "no empty captions")
    r.check(len({i.sha256 for i in images.values()}) == len(images),
            "no duplicate images (by content hash)")
    r.check(all(i.mode == "RGB" for i in images.values()),
            "all images are RGB",
            str({i.image_id: i.mode for i in images.values() if i.mode != "RGB"}))
    r.check(len({row["instruction"] for row in rows}) == 1,
            "a single instruction is used everywhere")

    # ── Splits ───────────────────────────────────────────────────────────────
    if splits is not None:
        members = {name: {row["image_id"] for row in part}
                   for name, part in splits.items()}
        overlaps = [f"{a}&{b}={sorted(members[a] & members[b])}"
                    for i, a in enumerate(members)
                    for b in list(members)[i + 1:]
                    if members[a] & members[b]]
        r.check(not overlaps, "NO IMAGE LEAKAGE between splits", "; ".join(overlaps))

        covered = set().union(*members.values()) if members else set()
        r.check(covered == set(images),
                "every image is assigned to exactly one split")

        for name, part in splits.items():
            counts = defaultdict(set)
            for row in part:
                counts[row["disease"]].add(row["image_id"])
            sizes = {len(v) for v in counts.values()}
            expected = getattr(cfg.split, f"{name}_per_class")
            r.check(len(counts) == n_classes and sizes == {expected},
                    f"{name} split is balanced ({expected} image(s) per class)",
                    str({k: len(v) for k, v in counts.items()}))

    # ── Quality warnings (never block) ───────────────────────────────────────
    lengths = [row["n_words"] for row in rows]
    if lengths:
        r.warn(min(lengths) >= 15,
               f"shortest caption is {min(lengths)} words")
    unresolved = [i.image_id for i in images.values() if i.provenance is None]
    if any(i.provenance for i in images.values()):
        r.warn(not unresolved,
               f"{len(unresolved)} image(s) not matched to the source dataset")

    return r

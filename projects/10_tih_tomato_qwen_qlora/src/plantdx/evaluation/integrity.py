"""Split-integrity check: no train image may appear in the eval split.

Reads ONLY `train.jsonl` and the target split's jsonl (default `test.jsonl`) from
the frozen dataset directory — never writes to either file. This is a read-only
regression check on an already-frozen artifact; it does not re-derive splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from plantdx.core.exceptions import DerivationError, InvariantViolation
from plantdx.utils.io import read_jsonl

_MAX_REPORTED_OVERLAPS = 20


@dataclass(frozen=True)
class IntegrityReport:
    """Outcome of the train/eval-split leakage check."""

    train_image_count: int
    eval_image_count: int
    eval_row_count: int
    overlap_count: int
    overlap_sample: tuple[str, ...]  # up to _MAX_REPORTED_OVERLAPS offending paths


def check_split_integrity(dataset_dir: str | Path, split: str) -> IntegrityReport:
    """Verify no image in `train.jsonl` also appears in `<split>.jsonl`.

    Raises :class:`DerivationError` if either file is missing, or
    :class:`InvariantViolation` (fail closed) if any image path overlaps.
    """
    root = Path(dataset_dir)
    train_path = root / "train.jsonl"
    eval_path = root / f"{split}.jsonl"
    for path in (train_path, eval_path):
        if not path.is_file():
            raise DerivationError(
                f"evaluation dataset file not found: {path}. The training data "
                f"pipeline (`plantdx prepare-training`) must be run first; this "
                f"stage never regenerates it."
            )

    train_images = _image_paths(train_path)
    eval_rows = list(read_jsonl(eval_path))
    eval_images = {str(row["image"]) for row in eval_rows}

    overlap = sorted(train_images & eval_images)
    report = IntegrityReport(
        train_image_count=len(train_images),
        eval_image_count=len(eval_images),
        eval_row_count=len(eval_rows),
        overlap_count=len(overlap),
        overlap_sample=tuple(overlap[:_MAX_REPORTED_OVERLAPS]),
    )
    if overlap:
        sample = ", ".join(report.overlap_sample)
        remaining = len(overlap) - _MAX_REPORTED_OVERLAPS
        more = f" (+{remaining} more)" if remaining > 0 else ""
        raise InvariantViolation(
            f"{len(overlap)} image(s) in train.jsonl also appear in {split}.jsonl "
            f"-- the eval split is contaminated: {sample}{more}"
        )
    return report


def _image_paths(path: Path) -> set[str]:
    return {str(row["image"]) for row in read_jsonl(path)}

"""Disease-label extraction from free text, and classification metrics.

Extraction is deterministic keyword/phrase matching against a lexicon built
from the frozen DKB's own `class_label`/`common_name` fields (read-only; no new
facts invented, no DKB edit). Classification metrics use scikit-learn's
reference implementations, never hand-rolled formulas.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from plantdx.core.exceptions import DerivationError

_DEFAULT_DKB = Path("knowledge_base/dkb.json")
_UNKNOWN = "unclassified"


@dataclass(frozen=True)
class DiseaseLexicon:
    """Disease id -> surface-form phrases, sorted longest-first for matching."""

    phrases_by_disease: dict[str, tuple[str, ...]]
    disease_ids: tuple[str, ...]

    def all_phrases_longest_first(self) -> list[tuple[str, str]]:
        """Return (phrase_lower, disease_id) pairs, longest phrase first."""
        pairs = [
            (phrase.lower(), disease_id)
            for disease_id, phrases in self.phrases_by_disease.items()
            for phrase in phrases
        ]
        return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


def build_lexicon(crop: str, *, dkb_path: str | Path = _DEFAULT_DKB) -> DiseaseLexicon:
    """Build a keyword lexicon for `crop` from the DKB's display-name fields."""
    path = Path(dkb_path)
    if not path.is_file():
        raise DerivationError(f"DKB not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    phrases_by_disease: dict[str, tuple[str, ...]] = {}
    for entry in data["diseases"]:
        if entry["crop"] != crop:
            continue
        phrases = {
            str(entry["class_label"]).strip(),
            _strip_crop_suffix(str(entry["disease"])),
            str(entry.get("common_name", "")).strip(),
        }
        phrases.update(_aliases(entry["id"]))
        phrases.discard("")
        phrases_by_disease[entry["id"]] = tuple(sorted(phrases))
    return DiseaseLexicon(
        phrases_by_disease=phrases_by_disease,
        disease_ids=tuple(sorted(phrases_by_disease)),
    )


def _strip_crop_suffix(disease_field: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", disease_field).strip()


def _aliases(disease_id: str) -> set[str]:
    extra = {
        "tomato_spider_mites": {"spider mite", "spider mites", "two-spotted spider mite"},
        "tomato_mosaic_virus": {"ToMV", "mosaic virus"},
        "tomato_yellow_leaf_curl_virus": {"TYLCV", "yellow leaf curl"},
        "tomato_septoria_leaf_spot": {"septoria"},
    }
    return extra.get(disease_id, set())


def extract_disease_id(text: str, lexicon: DiseaseLexicon) -> str:
    """Return the longest lexicon phrase found in `text`, or ``"unclassified"``.

    Longest-phrase-first matching avoids "spot" alone matching both "Bacterial
    Spot" and "Target Spot"; the full phrase must be present.
    """
    lowered = text.lower()
    for phrase, disease_id in lexicon.all_phrases_longest_first():
        if phrase.lower() in lowered:
            return disease_id
    return _UNKNOWN


@dataclass(frozen=True)
class ClassificationMetrics:
    """Standard classification metrics for extracted vs. ground-truth labels."""

    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    top1_accuracy: float  # identical to `accuracy` here (single-label extraction)
    unclassified_count: int
    sample_count: int


def compute_classification_metrics(
    predictions: Sequence[str], targets: Sequence[str], labels: Sequence[str]
) -> ClassificationMetrics:
    """Compute sklearn-backed classification metrics for predicted vs. true ids.

    `labels` fixes the label set (and its order) so metrics are stable even if a
    class is entirely absent from one side.
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        precision_recall_fscore_support,
    )

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must be the same length")

    label_list = list(labels)
    accuracy = sum(p == t for p, t in zip(predictions, targets, strict=True)) / len(targets)
    balanced = float(balanced_accuracy_score(targets, predictions))

    def _prf(average: str) -> tuple[float, float, float]:
        p, r, f, _ = precision_recall_fscore_support(
            targets, predictions, labels=label_list, average=average, zero_division=0
        )
        return float(p), float(r), float(f)

    p_macro, r_macro, f_macro = _prf("macro")
    p_weighted, r_weighted, f_weighted = _prf("weighted")
    p_micro, r_micro, f_micro = _prf("micro")

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced,
        precision_macro=p_macro,
        recall_macro=r_macro,
        f1_macro=f_macro,
        precision_weighted=p_weighted,
        recall_weighted=r_weighted,
        f1_weighted=f_weighted,
        precision_micro=p_micro,
        recall_micro=r_micro,
        f1_micro=f_micro,
        top1_accuracy=accuracy,
        unclassified_count=sum(1 for p in predictions if p == _UNKNOWN),
        sample_count=len(targets),
    )

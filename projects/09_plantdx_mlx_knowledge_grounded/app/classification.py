"""Robust disease-id extraction for the demo.

The frozen production extractor (``plantdx.evaluation.classification``) matches
exact lowercased phrases from the DKB's display fields. That's correct for the
evaluation corpus but a little brittle on free-form model text: it misses
one-word/spelling variants (``dieback`` vs ``die back``, ``sooty mold`` vs
``sooty mould``) and produces a false ``unclassified``.

This module wraps it: first try production extraction (unchanged, authoritative);
only if that says ``unclassified`` do we retry with a normalized, alias-aware,
crop-scoped matcher built from the same DKB fields. Production behavior — and the
evaluation numbers — are never altered.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from app.utils import DKB_PATH, strip_crop_suffix

UNCLASSIFIED = "unclassified"

# Curated extra surface forms the DKB fields don't spell out literally. Keyed by
# disease id; every phrase is normalized the same way as the text before matching.
_ALIASES: dict[str, tuple[str, ...]] = {
    "mango_die_back": ("dieback", "die back", "die-back", "twig blight"),
    "mango_sooty_mould": ("sooty mould", "sooty mold", "sooty"),
    "mango_gall_midge": ("gall midge", "leaf gall", "gall"),
    "mango_cutting_weevil": ("cutting weevil", "leaf-cutting weevil", "weevil"),
    "mango_bacterial_canker": ("bacterial canker", "bacterial black spot", "black spot"),
    "mango_powdery_mildew": ("powdery mildew",),
    "mango_anthracnose": ("anthracnose",),
    "tomato_spider_mites": ("spider mite", "spider mites", "two-spotted spider mite"),
    "tomato_mosaic_virus": ("mosaic virus", "mosaic", "tomv"),
    "tomato_yellow_leaf_curl_virus": ("yellow leaf curl", "tylcv"),
    "tomato_septoria_leaf_spot": ("septoria leaf spot", "septoria"),
    "tomato_target_spot": ("target spot",),
    "tomato_leaf_mold": ("leaf mold", "leaf mould"),
    "tomato_early_blight": ("early blight",),
    "tomato_late_blight": ("late blight",),
    "tomato_bacterial_spot": ("bacterial spot", "bacterial leaf spot"),
}

# Phrases this short are too ambiguous to match on their own.
_MIN_PHRASE_LEN = 4


def classify(caption: str, crop: str) -> str:
    """Return the best disease id for ``caption`` within ``crop``, or ``unclassified``.

    Never returns a disease id from a different crop.
    """
    from plantdx.evaluation.classification import build_lexicon, extract_disease_id

    lexicon = build_lexicon(crop, dkb_path=DKB_PATH)
    primary = extract_disease_id(caption, lexicon)
    if primary != UNCLASSIFIED:
        return primary
    return _fallback_match(caption, crop)


def _fallback_match(caption: str, crop: str) -> str:
    text = _normalize(caption)
    text_ns = text.replace(" ", "")
    if not text:
        return UNCLASSIFIED
    # Longest phrase first so "septoria leaf spot" wins over "spot".
    for phrase, disease_id in _phrase_index(crop):
        if phrase in text or phrase.replace(" ", "") in text_ns:
            return disease_id
    return UNCLASSIFIED


@lru_cache(maxsize=4)
def _phrase_index(crop: str) -> tuple[tuple[str, str], ...]:
    """Normalized ``(phrase, disease_id)`` pairs for one crop, longest first."""
    pairs: list[tuple[str, str]] = []
    for entry in _dkb_diseases():
        if str(entry.get("crop")) != crop:
            continue
        disease_id = str(entry["id"])
        for raw in _surface_forms(entry, disease_id):
            norm = _normalize(raw)
            if len(norm) >= _MIN_PHRASE_LEN:
                pairs.append((norm, disease_id))
    # Deduplicate, then sort by phrase length descending (longest, most specific first).
    unique = sorted(set(pairs), key=lambda p: len(p[0]), reverse=True)
    return tuple(unique)


def _surface_forms(entry: dict[str, Any], disease_id: str) -> set[str]:
    forms: set[str] = {
        str(entry.get("class_label", "")),
        strip_crop_suffix(str(entry.get("disease", ""))),
    }
    forms.update(_split_parentheticals(str(entry.get("common_name", ""))))
    forms.update(_ALIASES.get(disease_id, ()))
    forms.discard("")
    return forms


def _split_parentheticals(text: str) -> set[str]:
    """``"Sooty mould (sooty mold)"`` -> ``{"Sooty mould", "sooty mold"}``."""
    if not text:
        return set()
    outside = re.sub(r"\([^)]*\)", " ", text)
    inside = re.findall(r"\(([^)]*)\)", text)
    # Split alternatives on "/" too ("bacterial black spot / bacterial canker").
    parts = [outside, *inside]
    out: set[str] = set()
    for part in parts:
        out.update(p.strip() for p in part.split("/"))
    return {p for p in out if p}


def _normalize(text: str) -> str:
    """Lowercase, turn non-alphanumerics into single spaces, collapse, strip."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@lru_cache(maxsize=1)
def _dkb_diseases() -> tuple[dict[str, Any], ...]:
    if not DKB_PATH.is_file():
        return ()
    data = json.loads(DKB_PATH.read_text(encoding="utf-8"))
    return tuple(data.get("diseases", []))

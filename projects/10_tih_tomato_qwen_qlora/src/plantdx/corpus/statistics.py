"""Deterministic caption-corpus statistics (``statistics.json``).

Includes a lexical-diversity battery (distinct-1/2/3, entropy, sentence-opener
diversity, vocabulary utilization, mean reuse) so a release candidate's linguistic
quality is measurable and version-controlled. All values are pure functions of the
caption text — no randomness, no external models.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from plantdx.corpus.models import Corpus

_WORD = re.compile(r"[a-z]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(toks: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def _distinct(counter: Counter[Any]) -> float:
    total = sum(counter.values())
    return round(len(counter) / total, 4) if total else 0.0


def compute(corpus: Corpus, validation_status: str) -> dict[str, Any]:
    """Return the statistics document (all values deterministic)."""
    captions = corpus.captions
    by_disease: dict[str, int] = {}
    by_family: dict[str, int] = {}
    by_register: dict[str, int] = {}
    by_confidence: dict[str, int] = {}
    g1: Counter[tuple[str, ...]] = Counter()
    g2: Counter[tuple[str, ...]] = Counter()
    g3: Counter[tuple[str, ...]] = Counter()
    openers: Counter[str] = Counter()
    total_tokens = 0
    hedged = 0
    for c in captions:
        by_disease[c.disease_id] = by_disease.get(c.disease_id, 0) + 1
        by_family[c.family] = by_family.get(c.family, 0) + 1
        by_register[c.register] = by_register.get(c.register, 0) + 1
        by_confidence[c.confidence] = by_confidence.get(c.confidence, 0) + 1
        hedged += int(c.hedged)
        toks = _tokens(c.text)
        total_tokens += len(toks)
        g1.update(_ngrams(toks, 1))
        g2.update(_ngrams(toks, 2))
        g3.update(_ngrams(toks, 3))
        openers[" ".join(toks[:3])] += 1

    unigram_total = sum(g1.values())
    entropy = (
        round(-sum((n / unigram_total) * math.log2(n / unigram_total) for n in g1.values()), 4)
        if unigram_total
        else 0.0
    )
    disease_counts: list[int] = list(by_disease.values())
    min_per_disease: int = min(disease_counts) if disease_counts else 0
    max_per_disease: int = max(disease_counts) if disease_counts else 0
    imbalance_ratio = round(max_per_disease / min_per_disease, 2) if min_per_disease else 0.0

    return {
        "content_hash": corpus.provenance.get("content_hash", ""),
        "validation_status": validation_status,
        "caption_count": len(captions),
        "disease_count": len(by_disease),
        "hedged_count": hedged,
        "total_tokens": total_tokens,
        "mean_tokens_per_caption": round(total_tokens / len(captions), 2) if captions else 0.0,
        "diversity": {
            "distinct_1": _distinct(g1),
            "distinct_2": _distinct(g2),
            "distinct_3": _distinct(g3),
            "unique_unigrams": len(g1),
            "unique_bigrams": len(g2),
            "unique_trigrams": len(g3),
            "lexical_entropy_bits": entropy,
            "mean_token_reuse": round(unigram_total / len(g1), 2) if g1 else 0.0,
            "distinct_openers": len(openers),
            "opener_diversity": round(len(openers) / len(captions), 4) if captions else 0.0,
            "top_opener_share": round(openers.most_common(1)[0][1] / len(captions), 4)
            if captions
            else 0.0,
        },
        "balance": {
            "min_per_disease": min_per_disease,
            "max_per_disease": max_per_disease,
            "imbalance_ratio": imbalance_ratio,
        },
        "by_disease": {k: by_disease[k] for k in sorted(by_disease)},
        "by_family": {k: by_family[k] for k in sorted(by_family)},
        "by_register": {k: by_register[k] for k in sorted(by_register)},
        "by_confidence": {k: by_confidence[k] for k in sorted(by_confidence)},
    }

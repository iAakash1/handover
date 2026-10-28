"""Response-quality heuristics: length, sentences, fluency, redundancy, diversity.

No external grammar-checker dependency (``language-tool-python`` needs a Java
server and is not part of the offline-reproducible ``[eval]`` extra); fluency is
a small, deterministic, documented heuristic instead of an approximated "grammar
score" from a tool this pipeline does not actually run.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from plantdx.evaluation.text_metrics import tokenize

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class ResponseQuality:
    """Per-response quality measurements."""

    word_count: int
    sentence_count: int
    fluency_score: float  # 0..1 heuristic (capitalization, punctuation, spacing)
    redundancy_rate: float  # fraction of adjacent word-bigram repeats
    repetition_rate: float  # fraction of sentences that duplicate an earlier one
    lexical_diversity: float  # type-token ratio within this one response


def score_response_quality(text: str) -> ResponseQuality:
    """Score one response's length, fluency, redundancy, and diversity."""
    words = _WORD_RE.findall(text)
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    tokens = tokenize(text)

    return ResponseQuality(
        word_count=len(words),
        sentence_count=len(sentences),
        fluency_score=_fluency_score(text, sentences),
        redundancy_rate=_redundancy_rate(tokens),
        repetition_rate=_repetition_rate(sentences),
        lexical_diversity=(len(set(tokens)) / len(tokens)) if tokens else 0.0,
    )


def summarize_response_quality(rows: Sequence[ResponseQuality]) -> dict[str, float]:
    """Corpus-level means of every :class:`ResponseQuality` field."""
    if not rows:
        return {
            "avg_word_count": 0.0,
            "avg_sentence_count": 0.0,
            "avg_fluency_score": 0.0,
            "avg_redundancy_rate": 0.0,
            "avg_repetition_rate": 0.0,
            "avg_lexical_diversity": 0.0,
        }
    n = len(rows)
    return {
        "avg_word_count": sum(r.word_count for r in rows) / n,
        "avg_sentence_count": sum(r.sentence_count for r in rows) / n,
        "avg_fluency_score": sum(r.fluency_score for r in rows) / n,
        "avg_redundancy_rate": sum(r.redundancy_rate for r in rows) / n,
        "avg_repetition_rate": sum(r.repetition_rate for r in rows) / n,
        "avg_lexical_diversity": sum(r.lexical_diversity for r in rows) / n,
    }


def _fluency_score(text: str, sentences: list[str]) -> float:
    """A bounded [0,1] heuristic combining five cheap well-formedness checks.

    Starts capitalized, ends with terminal punctuation, no double spaces, has
    at least one sentence, and no wildly long "words" (a sign of missing
    spaces / garbled generation).
    """
    if not text.strip():
        return 0.0
    checks = [
        text[:1].isupper(),
        text.rstrip().endswith((".", "!", "?")),
        "  " not in text,
        len(sentences) > 0,
        all(len(w) <= 25 for w in _WORD_RE.findall(text)),
    ]
    return sum(checks) / len(checks)


def _redundancy_rate(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    repeats = sum(1 for a, b in pairwise(tokens) if a == b)
    return repeats / (len(tokens) - 1)


def _repetition_rate(sentences: list[str]) -> float:
    if len(sentences) < 2:
        return 0.0
    seen: set[str] = set()
    duplicates = 0
    for sentence in sentences:
        normalized = sentence.lower().strip()
        if normalized in seen:
            duplicates += 1
        seen.add(normalized)
    return duplicates / len(sentences)

"""Caption-quality text metrics: BLEU, ROUGE-L, METEOR, CIDEr, BERTScore, and more.

Every metric uses its official reference implementation (never an approximation):
BLEU-1..4 and CIDEr via ``pycocoevalcap`` (the MS-COCO caption evaluation
toolkit), ROUGE-L via Google Research's ``rouge-score``, METEOR via
``nltk.translate.meteor_score`` + WordNet, BERTScore via the ``bert-score``
package. All are lazy-imported (the `--stage analyze` environment; never
required where mlx-vlm runs) and fail closed with an actionable message,
pointing at ``make install-eval`` / ``scripts/setup_eval_env.sh``, if a
resource is missing -- this module never downloads anything itself.

CIDEr's TF-IDF weighting needs the *whole* reference corpus to produce a
meaningful (non-degenerate) score, so :func:`compute_text_metrics` always scores
the full batch of samples together, not one pair at a time.
"""

from __future__ import annotations

import contextlib
import io
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from plantdx.core.exceptions import PlantDxError

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBA_HINT = "numpy.core.multiarray failed to import"


@dataclass(frozen=True)
class TextMetricsPerSample:
    """Per-sample scores for one (hypothesis, reference) pair."""

    id: str
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    rouge_l: float
    meteor: float
    cider: float
    bertscore_f1: float
    token_overlap: float
    sentence_similarity: float


@dataclass(frozen=True)
class TextMetricsCorpus:
    """Corpus-level aggregate scores over the whole evaluated batch."""

    sample_count: int
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    rouge_l: float
    meteor: float
    cider: float
    bertscore_f1: float
    token_overlap: float
    sentence_similarity: float
    distinct_1: float
    distinct_2: float


def tokenize(text: str) -> list[str]:
    """Deterministic, dependency-free tokenizer: lowercase alnum tokens."""
    return _TOKEN_RE.findall(text.lower())


def compute_text_metrics(
    pairs: Sequence[tuple[str, str, str]],
) -> tuple[list[TextMetricsPerSample], TextMetricsCorpus]:
    """Score every (id, hypothesis, reference) triple; return (per-sample, corpus).

    Raises :class:`PlantDxError` if a required metrics library or resource
    (WordNet, the BERTScore backbone model) is missing.
    """
    if not pairs:
        raise PlantDxError("compute_text_metrics: no samples to score")

    ids = [p[0] for p in pairs]
    hyps = [p[1] for p in pairs]
    refs = [p[2] for p in pairs]

    bleu_corpus, bleu_per_sample = _compute_bleu(ids, hyps, refs)
    cider_corpus, cider_per_sample = _compute_cider(ids, hyps, refs)
    rouge_per_sample = [_compute_rouge_l(h, r) for h, r in zip(hyps, refs, strict=True)]
    meteor_per_sample = [_compute_meteor(h, r) for h, r in zip(hyps, refs, strict=True)]
    bertscore_per_sample = _compute_bertscore(hyps, refs)
    overlap_per_sample = [_token_overlap(h, r) for h, r in zip(hyps, refs, strict=True)]
    similarity_per_sample = _sentence_similarity(hyps, refs)

    per_sample = [
        TextMetricsPerSample(
            id=ids[i],
            bleu1=bleu_per_sample[0][i],
            bleu2=bleu_per_sample[1][i],
            bleu3=bleu_per_sample[2][i],
            bleu4=bleu_per_sample[3][i],
            rouge_l=rouge_per_sample[i],
            meteor=meteor_per_sample[i],
            cider=cider_per_sample[i],
            bertscore_f1=bertscore_per_sample[i],
            token_overlap=overlap_per_sample[i],
            sentence_similarity=similarity_per_sample[i],
        )
        for i in range(len(ids))
    ]
    corpus = TextMetricsCorpus(
        sample_count=len(ids),
        bleu1=bleu_corpus[0],
        bleu2=bleu_corpus[1],
        bleu3=bleu_corpus[2],
        bleu4=bleu_corpus[3],
        rouge_l=_mean(rouge_per_sample),
        meteor=_mean(meteor_per_sample),
        cider=cider_corpus,
        bertscore_f1=_mean(bertscore_per_sample),
        token_overlap=_mean(overlap_per_sample),
        sentence_similarity=_mean(similarity_per_sample),
        distinct_1=_distinct_n(hyps, 1),
        distinct_2=_distinct_n(hyps, 2),
    )
    return per_sample, corpus


# --------------------------------------------------------------------------- #
# Official-library integrations (each lazy-imported, each fails closed)
# --------------------------------------------------------------------------- #


def _compute_bleu(
    ids: list[str], hyps: list[str], refs: list[str]
) -> tuple[list[float], list[list[float]]]:
    """BLEU-1..4 via pycocoevalcap's official Bleu scorer (cumulative n-gram)."""
    try:
        from pycocoevalcap.bleu.bleu import Bleu
    except ImportError as exc:
        raise _missing("pycocoevalcap", "BLEU-1..4") from exc

    gts = {i: [r] for i, r in zip(ids, refs, strict=True)}
    res = {i: [h] for i, h in zip(ids, hyps, strict=True)}
    with contextlib.redirect_stdout(io.StringIO()):
        corpus_scores, per_sample_scores = Bleu(4).compute_score(gts, res)
    # per_sample_scores[n] is a list aligned to `ids`' insertion order (dict is
    # ordered in Python 3.7+, and gts/res were built from `ids` in order).
    per_sample = [[float(s) for s in order_scores] for order_scores in per_sample_scores]
    return [float(s) for s in corpus_scores], per_sample


def _compute_cider(ids: list[str], hyps: list[str], refs: list[str]) -> tuple[float, list[float]]:
    """CIDEr via pycocoevalcap's official Cider scorer.

    Scored over the FULL corpus at once (its TF-IDF weighting is meaningless on
    a single pair).
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError as exc:
        raise _missing("pycocoevalcap", "CIDEr") from exc

    gts = {i: [r] for i, r in zip(ids, refs, strict=True)}
    res = {i: [h] for i, h in zip(ids, hyps, strict=True)}
    with contextlib.redirect_stdout(io.StringIO()):
        corpus_score, per_sample_scores = Cider().compute_score(gts, res)
    return float(corpus_score), [float(s) for s in per_sample_scores]


_ROUGE_SCORER_CACHE: Any = None


def _compute_rouge_l(hyp: str, ref: str) -> float:
    """ROUGE-L F-measure via Google Research's official rouge-score.

    Reuses one module-level ``RougeScorer`` across all per-sample calls in a
    batch (construction loads stemmer resources; the scorer itself is
    stateless per call).
    """
    global _ROUGE_SCORER_CACHE
    if _ROUGE_SCORER_CACHE is None:
        try:
            from rouge_score import rouge_scorer
        except ImportError as exc:
            raise _missing("rouge-score", "ROUGE-L") from exc
        _ROUGE_SCORER_CACHE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(_ROUGE_SCORER_CACHE.score(ref, hyp)["rougeL"].fmeasure)


def _compute_meteor(hyp: str, ref: str) -> float:
    """METEOR via nltk.translate.meteor_score + WordNet (official NLTK impl)."""
    try:
        from nltk.translate.meteor_score import single_meteor_score
    except ImportError as exc:
        raise _missing("nltk", "METEOR") from exc

    try:
        return float(single_meteor_score(tokenize(ref), tokenize(hyp)))
    except LookupError as exc:  # WordNet corpus not downloaded
        raise PlantDxError(
            "METEOR requires the NLTK WordNet corpus, which is not downloaded. "
            "Run `make install-eval` (scripts/setup_eval_env.sh) once to cache it."
        ) from exc


def _compute_bertscore(hyps: list[str], refs: list[str]) -> list[float]:
    """BERTScore F1 via the official bert-score package, batched over the set."""
    try:
        import bert_score
    except ImportError as exc:
        raise _missing("bert-score", "BERTScore") from exc

    try:
        _, _, f1 = bert_score.score(hyps, refs, lang="en", verbose=False)
    except Exception as exc:  # backbone load failures surface in many shapes
        if _NUMBA_HINT in str(exc):
            raise PlantDxError(
                "BERTScore failed to load its backbone model because of a "
                "pre-existing numba/NumPy ABI conflict in this environment "
                '(unrelated to PlantDx). Fix: `pip install -U "numba>=0.59" '
                '"llvmlite>=0.42"`, or run scripts/setup_eval_env.sh in a '
                "fresh virtualenv. See docs/EVALUATION.md#troubleshooting."
            ) from exc
        raise PlantDxError(
            f"BERTScore backbone model is not available: {exc}. Run "
            f"`make install-eval` (scripts/setup_eval_env.sh) to cache it."
        ) from exc
    return [float(v) for v in f1.tolist()]


def _missing(package: str, metric: str) -> PlantDxError:
    return PlantDxError(
        f"{metric} requires the '{package}' package, which is not installed in "
        f"this environment. Run `make install-eval` (scripts/setup_eval_env.sh) "
        f"-- see docs/EVALUATION.md for the two-stage evaluation environment split."
    )


# --------------------------------------------------------------------------- #
# Lightweight, dependency-minimal metrics (no external "official" library named)
# --------------------------------------------------------------------------- #


def _token_overlap(hyp: str, ref: str) -> float:
    """Fraction of reference tokens (multiset) also present in the hypothesis."""
    ref_tokens = tokenize(ref)
    if not ref_tokens:
        return 0.0
    hyp_tokens = tokenize(hyp)
    hyp_counts: dict[str, int] = {}
    for tok in hyp_tokens:
        hyp_counts[tok] = hyp_counts.get(tok, 0) + 1
    matched = 0
    for tok in ref_tokens:
        if hyp_counts.get(tok, 0) > 0:
            hyp_counts[tok] -= 1
            matched += 1
    return matched / len(ref_tokens)


def _sentence_similarity(hyps: list[str], refs: list[str]) -> list[float]:
    """TF-IDF cosine similarity between each hypothesis and its reference.

    Not a neural embedding similarity (that role is BERTScore's, computed
    separately with the official library); this is a lexical-overlap similarity
    fit over the batch's own vocabulary, reported explicitly as such.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = hyps + refs
    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    matrix = vectorizer.fit_transform(corpus)
    n = len(hyps)
    hyp_vectors, ref_vectors = matrix[:n], matrix[n:]
    sims = cosine_similarity(hyp_vectors, ref_vectors)
    return [float(sims[i, i]) for i in range(n)]


def _distinct_n(texts: list[str], n: int) -> float:
    """Corpus-level distinct-N: unique n-grams / total n-grams across all texts."""
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        toks = tokenize(text)
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

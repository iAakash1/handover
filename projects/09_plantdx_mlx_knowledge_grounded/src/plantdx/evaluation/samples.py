"""50 deterministic sample comparisons: base vs. fine-tuned, as a markdown table."""

from __future__ import annotations

from dataclasses import dataclass

from plantdx.utils.hashing import sha256_hex

_DEFAULT_SAMPLE_COUNT = 50


@dataclass(frozen=True)
class SampleComparison:
    """One sample's full comparison, with an automatic, explainable verdict."""

    image_path: str
    ground_truth_disease: str
    instruction: str
    ground_truth_answer: str
    base_answer: str
    finetuned_answer: str
    base_bleu1: float
    finetuned_bleu1: float
    winner: str  # "base" | "finetuned" | "tie"
    reason: str


def select_sample_indices(
    total: int, *, seed: int, count: int = _DEFAULT_SAMPLE_COUNT
) -> list[int]:
    """Deterministically select up to `count` sample indices from `[0, total)`.

    A pure function of `(seed, total)` -- the same seed always yields the same
    indices, in the same order, with no dependency on RNG state.
    """
    if total <= 0:
        return []
    scored = sorted(range(total), key=lambda i: sha256_hex(str(seed), str(i)))
    return sorted(scored[: min(count, total)])


def build_sample_comparison(
    *,
    image_path: str,
    ground_truth_disease: str,
    instruction: str,
    ground_truth_answer: str,
    base_answer: str,
    finetuned_answer: str,
    base_bleu1: float,
    finetuned_bleu1: float,
) -> SampleComparison:
    """Build one comparison row with an automatic winner + a stated reason."""
    winner, reason = _judge(base_bleu1, finetuned_bleu1)
    return SampleComparison(
        image_path=image_path,
        ground_truth_disease=ground_truth_disease,
        instruction=instruction,
        ground_truth_answer=ground_truth_answer,
        base_answer=base_answer,
        finetuned_answer=finetuned_answer,
        base_bleu1=base_bleu1,
        finetuned_bleu1=finetuned_bleu1,
        winner=winner,
        reason=reason,
    )


def _judge(base_bleu1: float, finetuned_bleu1: float, *, margin: float = 0.02) -> tuple[str, str]:
    """Pick a winner from BLEU-1 vs. the ground truth, with a stated margin.

    BLEU-1 (unigram overlap with the frozen ground-truth caption) is the
    deterministic, reproducible tie-breaker reported here; a human reviewer
    reading the table can judge holistically from the printed text of both
    answers, which is why the full text of both is always included alongside.
    """
    diff = finetuned_bleu1 - base_bleu1
    if abs(diff) < margin:
        return "tie", (
            f"BLEU-1 within {margin:.2f} of each other ({base_bleu1:.2f} vs {finetuned_bleu1:.2f})"
        )
    if diff > 0:
        return "finetuned", f"fine-tuned BLEU-1 {finetuned_bleu1:.2f} > base {base_bleu1:.2f}"
    return "base", f"base BLEU-1 {base_bleu1:.2f} > fine-tuned {finetuned_bleu1:.2f}"


def render_markdown_table(rows: list[SampleComparison]) -> str:
    """Render the sample comparisons as one markdown table."""
    lines = [
        "| Image | Ground truth disease | Question | Ground truth answer | "
        "Base answer | Fine-tuned answer | Winner | Reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        image = _escape(row.image_path)
        truth = _escape(row.ground_truth_disease)
        question = _escape(row.instruction)
        gt = _escape(row.ground_truth_answer)
        base = _escape(row.base_answer)
        ft = _escape(row.finetuned_answer)
        reason = _escape(row.reason)
        lines.append(
            f"| {image} | {truth} | {question} | {gt} | {base} | {ft} | {row.winner} | {reason} |"
        )
    return "\n".join(lines) + "\n"


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")

"""
Stage 9 — Evaluation.

FOUR QUESTIONS, FOUR MEASUREMENTS
    1. Does it write the kind of sentence our annotators write?
       BLEU-1..4 and ROUGE-L against ALL FIVE references. Multi-reference by
       construction: every image has five valid answers, so scoring against one
       would punish the model for choosing a different correct description.

    2. Is the description class-appropriate?
       The validated captions never name a disease (annotation guideline 5.3), so
       the model cannot be asked to classify without inventing supervision. Instead
       the generated text is compared to the TRAINING captions of each class and
       the closest class is taken as an implicit prediction. No new task, no
       modified caption, and still a number.

    3. Is it reciting instead of describing?
       With 18 training images, memorisation is the primary risk. Any output with
       high word overlap against a training caption is reported.

    4. Is it reproducing known caption defects?
       Six validated captions reference other images ("compared with the other
       leaves in this group"). If the model learned that phrasing it shows here.

READ THE CEILING BEFORE READING THE SCORE
    Measured on all 60 held-out human-written captions, question 2's metric
    recovers the true class only ~40% of the time (chance is 17%). That is a
    property of the data, not of the model: Bacterial spot and Early Blight, and
    Leaf Mold and Mosaic Virus, are described with the same visual vocabulary.
    Every report therefore prints the human ceiling computed on the same split. A
    model near the ceiling is doing as well as this supervision allows.

INPUT   a predictions JSON + the training split
OUTPUT  runs/<id>/evaluation/<split>_report.json + a printed report
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .io import read_json, write_json


# ── 1. Caption similarity ────────────────────────────────────────────────────

def caption_scores(generated: str, references: List[str]) -> Dict[str, float]:
    """BLEU-1..4 against the reference set; ROUGE-L max and mean over references."""
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_score import rouge_scorer as rouge_module

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    smooth = SmoothingFunction().method1
    hyp = generated.lower().split()
    refs = [r.lower().split() for r in references]

    scores = {
        f"bleu{n}": float(sentence_bleu(
            refs, hyp, weights=tuple([1.0 / n] * n + [0.0] * (4 - n)),
            smoothing_function=smooth))
        for n in (1, 2, 3, 4)
    }
    scorer = rouge_module.RougeScorer(["rougeL"], use_stemmer=True)
    rl = [scorer.score(r, generated)["rougeL"].fmeasure for r in references]
    scores["rougeL_max"] = float(max(rl))
    scores["rougeL_mean"] = float(np.mean(rl))
    return scores


# ── 2. Class retrieval ───────────────────────────────────────────────────────

class ClassRetriever:
    """
    Assigns a class to free text by comparing it to the training captions of each
    class and taking the highest MEAN similarity.

    Mean rather than nearest-neighbour: measured on all 60 held-out human
    captions, 1-NN recovers 18/60 and class-mean recovers 24/60. Averaging is less
    distracted by one incidentally similar sentence.
    """

    def __init__(self, train_rows: List[Dict], cfg: Config) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.texts = [r["caption"] for r in train_rows]
        self.labels = [r["disease"] for r in train_rows]
        self.classes = sorted(set(self.labels))
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, cfg.eval.retrieval_ngram_max),
            sublinear_tf=True, stop_words=cfg.eval.retrieval_stop_words)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def predict(self, text: str) -> Dict:
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0]
        per_class = {
            cls: float(np.mean([s for s, lab in zip(sims, self.labels) if lab == cls]))
            for cls in self.classes
        }
        predicted = max(per_class, key=per_class.get)
        return {"predicted": predicted, "similarity": per_class[predicted],
                "per_class_mean": per_class}

    def human_ceiling(self, rows: List[Dict]) -> Dict:
        """Score the split's own validated captions — the achievable maximum."""
        correct = sum(self.predict(r["caption"])["predicted"] == r["disease"]
                      for r in rows)
        return {"correct": correct, "total": len(rows),
                "value": correct / len(rows) if rows else 0.0}


# ── 3 + 4. Safety checks ─────────────────────────────────────────────────────

def word_overlap(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def memorisation_hit(generated: str, train_captions: List[str],
                     threshold: float) -> Optional[Dict]:
    best_score, best_text = 0.0, ""
    for caption in train_captions:
        score = word_overlap(generated, caption)
        if score > best_score:
            best_score, best_text = score, caption
    if best_score >= threshold:
        return {"overlap": round(best_score, 3), "training_caption": best_text}
    return None


def artefact_hits(generated: str, phrases: List[str]) -> List[str]:
    low = generated.lower()
    return [p for p in phrases if p in low]


# ── Report ───────────────────────────────────────────────────────────────────

def evaluate_predictions(cfg: Config, predictions_file: Path,
                         train_rows: List[Dict], split_rows: List[Dict],
                         retriever: Optional[ClassRetriever] = None) -> Dict:
    payload = read_json(predictions_file)
    retriever = retriever or ClassRetriever(train_rows, cfg)
    ceiling = retriever.human_ceiling(split_rows)
    train_captions = [r["caption"] for r in train_rows]

    per_image, aggregate = [], defaultdict(list)
    correct = 0
    memorised, artefacts = [], []

    for entry in payload["results"]:
        # The first sample is scored; the rest are kept for reading. Averaging
        # over samples would hide the variance that sampling is there to show.
        generated = entry["generated"][0]
        scores = caption_scores(generated, entry["references"])
        retrieval = retriever.predict(generated)
        hit = retrieval["predicted"] == entry["disease"]
        correct += int(hit)

        for k, v in scores.items():
            aggregate[k].append(v)

        mem = memorisation_hit(generated, train_captions,
                               cfg.eval.memorisation_threshold)
        if mem:
            memorised.append({"image_id": entry["image_id"], **mem})
        arts = artefact_hits(generated, cfg.eval.cross_image_phrases)
        if arts:
            artefacts.append({"image_id": entry["image_id"], "phrases": arts})

        per_image.append({
            "image_id": entry["image_id"],
            "true_class": entry["disease"],
            "predicted_class": retrieval["predicted"],
            "correct": hit,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "generated": generated,
            "n_words": len(generated.split()),
        })

    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in per_image:
        confusion[row["true_class"]][row["predicted_class"]] += 1

    total = len(per_image)
    return {
        "predictions_file": Path(predictions_file).name,
        "model": payload.get("model"),
        "n_images": total,
        "n_classes": len(retriever.classes),
        "retrieval_accuracy": f"{correct}/{total}",
        "retrieval_accuracy_value": correct / total if total else 0.0,
        "human_ceiling": ceiling,
        "chance": 1 / len(retriever.classes),
        "caption_scores_mean": {k: round(float(np.mean(v)), 4)
                                for k, v in aggregate.items()},
        "caption_length_mean": round(float(np.mean(
            [r["n_words"] for r in per_image])), 1) if per_image else 0.0,
        "memorised_captions": memorised,
        "cross_image_artefacts": artefacts,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "per_image": per_image,
    }


def render_report(report: Dict) -> str:
    lines = ["=" * 74,
             f"  {str(report['model']).upper()} — {report['n_images']} held-out images",
             "=" * 74, ""]

    ceiling = report.get("human_ceiling") or {}
    lines.append(f"  Class retrieval accuracy : {report['retrieval_accuracy']}")
    if ceiling:
        lines.append(f"    human ceiling          : {ceiling['correct']}/"
                     f"{ceiling['total']} ({ceiling['value']:.0%})  "
                     f"— the same metric on the validated captions themselves")
        lines.append(f"    chance                 : {report['chance']:.0%}")
        lines.append("    The captions never name a disease, so this measures how")
        lines.append("    distinguishable the CLASSES are in description-space.")
        lines.append("    Compare the model to the ceiling, never to 100%.")
    lines.append(f"  Mean generated length    : {report['caption_length_mean']} words")

    s = report["caption_scores_mean"]
    lines += ["", "  Caption similarity vs 5 references (progress signal, not a result)",
              f"    BLEU-1 {s['bleu1']:.3f}   BLEU-4 {s['bleu4']:.3f}   "
              f"ROUGE-L max {s['rougeL_max']:.3f}   mean {s['rougeL_mean']:.3f}",
              "", "  Per image"]
    for row in report["per_image"]:
        mark = "OK " if row["correct"] else "XX "
        lines.append(f"    {mark}{row['image_id']:<28} true={row['true_class']:<24}"
                     f"pred={row['predicted_class']}")

    off_diagonal = [(t, p, n) for t, preds in report["confusion"].items()
                    for p, n in preds.items() if p != t]
    if off_diagonal:
        lines += ["", "  Confusions"]
        lines += [f"    {t}  ->  {p}  ({n})" for t, p, n in off_diagonal]

    lines += ["", "  Safety checks"]
    if report["memorised_captions"]:
        lines.append(f"    MEMORISATION: {len(report['memorised_captions'])} output(s) "
                     f"match a training caption")
        lines += [f"      {m['image_id']}  overlap {m['overlap']}"
                  for m in report["memorised_captions"]]
    else:
        lines.append("    memorisation       : none detected")
    if report["cross_image_artefacts"]:
        lines.append(f"    ARTEFACTS: {len(report['cross_image_artefacts'])} output(s) "
                     f"use cross-image phrasing")
        lines += [f"      {a['image_id']}: {a['phrases']}"
                  for a in report["cross_image_artefacts"]]
    else:
        lines.append("    cross-image phrases: none detected")

    lines += ["", f"  Read the {report['n_images']} generated captions yourself — at "
                  f"this scale that outweighs every number above.", "=" * 74]
    return "\n".join(lines)


def compare(base: Dict, finetuned: Dict) -> str:
    """The before/after table — the headline result of the project."""
    lines = ["", "=" * 74, "  BASE  vs  FINE-TUNED", "=" * 74,
             f"    class retrieval : {base['retrieval_accuracy']:>9}   ->   "
             f"{finetuned['retrieval_accuracy']}   "
             f"(ceiling {finetuned['human_ceiling']['correct']}/"
             f"{finetuned['human_ceiling']['total']})"]
    for key in ("bleu1", "bleu4", "rougeL_max"):
        lines.append(f"    {key:<15} : {base['caption_scores_mean'][key]:>9.3f}   ->   "
                     f"{finetuned['caption_scores_mean'][key]:.3f}")
    lines.append(f"    mean length     : {base['caption_length_mean']:>9.1f}   ->   "
                 f"{finetuned['caption_length_mean']:.1f} words")
    lines.append("=" * 74)
    return "\n".join(lines)


def save_report(report: Dict, path: Path) -> None:
    write_json(report, path)

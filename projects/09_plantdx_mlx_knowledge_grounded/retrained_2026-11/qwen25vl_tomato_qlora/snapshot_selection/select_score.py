"""Score the validation answers of each snapshot with the evaluation's own parser and metrics.

Selection rule (fixed before any snapshot was scored): the highest validation macro-F1; on a tie, the
later snapshot. Only the validation split is used here; the test split is evaluated afterwards, once,
for the selected snapshot.

Usage (metrics env, from the project folder; reads selection/val_predictions/, writes selection/validation_scores.csv):
    python selection/select_score.py
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path

from plantdx.evaluation.classification import (
    build_lexicon,
    compute_classification_metrics,
    extract_disease_id,
)

OUT = Path("selection")


def main() -> None:
    lexicon = build_lexicon("tomato")
    results = []
    for f in sorted((OUT / "val_predictions").glob("iter_*.jsonl")):
        rows = [json.loads(line) for line in f.read_text().splitlines()]
        predicted = [extract_disease_id(r["finetuned_prediction"], lexicon) for r in rows]
        targets = [r["disease_id"] for r in rows]
        m = asdict(compute_classification_metrics(predicted, targets, lexicon.disease_ids))
        healthy = [p for p, t in zip(predicted, targets) if t == "tomato_healthy"]
        results.append({
            "iteration": int(f.stem.split("_")[1]),
            "samples": len(rows),
            "accuracy": round(m["accuracy"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "unclassified": m["unclassified_count"],
            "healthy_correct": sum(p == "tomato_healthy" for p in healthy),
            "healthy_total": len(healthy),
            "one_word_answers": sum(len(r["finetuned_prediction"].strip().rstrip(".")) <= 2 for r in rows),
        })
    results.sort(key=lambda r: r["iteration"])
    best = max(results, key=lambda r: (r["f1_macro"], r["iteration"]))
    with (OUT / "validation_scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]) + ["selected"])
        w.writeheader()
        for r in results:
            w.writerow({**r, "selected": r is best})
    for r in results:
        print(r, "<- selected" if r is best else "")


if __name__ == "__main__":
    main()

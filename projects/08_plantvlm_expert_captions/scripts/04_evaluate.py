"""
Stage 9 — Evaluation.

    python scripts/04_evaluate.py --split val
    python scripts/04_evaluate.py --split test --compare-baseline
    python scripts/04_evaluate.py --split val --run run_001_fold0

Reads a frozen predictions file and never re-runs the model, so a number can
never change because a report was regenerated.

INPUT   runs/<id>/predictions/<split>_*.json + data/train.jsonl
OUTPUT  runs/<id>/evaluation/<split>_report.json + a printed report
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import add_common_args, banner, load

from plantvlm.data import load_split
from plantvlm.evaluation import (ClassRetriever, compare, evaluate_predictions,
                                 render_report, save_report)
from plantvlm.training import Run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated captions")
    add_common_args(parser)
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--run", default=None, help="run name (default: latest)")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="also score <split>_base.json and print before/after")
    args = parser.parse_args()

    cfg = load(args)
    run = Run.resolve(cfg, args.run)

    train_rows = load_split(cfg, "train")
    split_rows = load_split(cfg, args.split)
    retriever = ClassRetriever(train_rows, cfg)
    ceiling = retriever.human_ceiling(split_rows)

    banner("Stage 9 — evaluation", cfg, (
        ("run", run.name),
        ("split", args.split),
        ("pool", f"{len(train_rows)} training captions, "
                 f"{len(retriever.classes)} classes"),
        ("ceiling", f"{ceiling['correct']}/{ceiling['total']} "
                    f"({ceiling['value']:.0%}) on this split's own captions"),
    ))

    ft_file = run.predictions / f"{args.split}_finetuned.json"
    if not ft_file.exists():
        print(f"ERROR: {ft_file} not found\n"
              f"       run: python scripts/03_infer.py --split {args.split}")
        sys.exit(1)

    report = evaluate_predictions(cfg, ft_file, train_rows, split_rows, retriever)
    print(render_report(report))
    combined = {"finetuned": report}

    if args.compare_baseline:
        base_file = run.predictions / f"{args.split}_base.json"
        if base_file.exists():
            base = evaluate_predictions(cfg, base_file, train_rows, split_rows,
                                        retriever)
            print(render_report(base))
            print(compare(base, report))
            combined["base"] = base
        else:
            print(f"\n  (no baseline at {base_file} — run "
                  f"03_infer.py --split {args.split} --baseline)")

    out_file = run.evaluation / f"{args.split}_report.json"
    save_report(combined, out_file)
    print(f"\n  report -> {out_file}")


if __name__ == "__main__":
    main()

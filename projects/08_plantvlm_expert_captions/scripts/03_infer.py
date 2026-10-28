"""
Stage 8 — Inference on held-out images (and baseline capture).

    python scripts/03_infer.py --split val               # fine-tuned
    python scripts/03_infer.py --split val --baseline    # untuned control
    python scripts/03_infer.py --split test --num-samples 3
    python scripts/03_infer.py --split val --run run_001_fold0

Run --baseline BEFORE training if you can: the same leaves described by the
untuned model are the control condition that makes the fine-tuned result
evidence rather than an assertion.

INPUT   data/<split>.jsonl + a run's best adapter
OUTPUT  runs/<id>/predictions/<split>_<base|finetuned>.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import add_common_args, banner, load

from plantvlm.data import load_split
from plantvlm.inference import predict
from plantvlm.models import get_adapter
from plantvlm.training import Run


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate captions for a split")
    add_common_args(parser)
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--run", default=None,
                        help="run directory or name (default: latest)")
    parser.add_argument("--baseline", action="store_true",
                        help="untuned base model — the control condition")
    parser.add_argument("--adapter", default=None,
                        help="explicit adapter path (default: <run>/checkpoints/best)")
    parser.add_argument("--greedy", action="store_true",
                        help="deterministic decoding instead of sampling")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="captions per image (sampling only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N images (smoke testing)")
    args = parser.parse_args()

    cfg = load(args)
    rows = load_split(cfg, args.split)

    try:
        run = Run.resolve(cfg, args.run)
    except FileNotFoundError as exc:
        if not args.baseline:
            print(f"ERROR: {exc}")
            sys.exit(1)
        # A baseline can be captured before any training has happened.
        run = Run.create(cfg, tag="baseline")
        run.write_lineage(cfg, {"note": "baseline capture, no training"})

    adapter_path = None
    if not args.baseline:
        adapter_path = Path(args.adapter) if args.adapter else run.best
        if not adapter_path.exists():
            print(f"ERROR: no adapter at {adapter_path}\n"
                  f"       train first, or run with --baseline")
            sys.exit(1)

    tag = "base" if args.baseline else "finetuned"
    banner(f"Stage 8 — inference ({tag})", cfg, (
        ("run", run.name),
        ("split", f"{args.split} — {len({r['image_id'] for r in rows})} images"),
        ("decode", "greedy" if args.greedy
                   else f"sampling T={cfg.generate.temperature} "
                        f"top_p={cfg.generate.top_p}"),
        ("adapter", str(adapter_path) if adapter_path else "none (base model)"),
    ))

    out_file = run.predictions / f"{args.split}_{tag}.json"
    predict(cfg, get_adapter(cfg), rows, out_file,
            adapter_path=adapter_path, greedy=args.greedy,
            num_samples=args.num_samples, limit=args.limit)

    print(f"\n  saved -> {out_file}")
    print(f"  Next: python scripts/04_evaluate.py --split {args.split}")


if __name__ == "__main__":
    main()

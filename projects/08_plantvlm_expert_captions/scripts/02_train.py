"""
Stage 7 — LoRA fine-tuning.

    python scripts/02_train.py                    # the run
    python scripts/02_train.py --set train.epochs=12
    python scripts/02_train.py --resume runs/run_001_fold0/checkpoints/epoch-3

Creates runs/run_NNN_foldK/ containing the frozen config, the exact JSONL used,
the lineage record, logs, and checkpoints. Nothing is overwritten between runs.

INPUT   data/{train,val}.jsonl
OUTPUT  a run directory with checkpoints/best/ as the artifact for inference
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import add_common_args, banner, load

from plantvlm.data import load_split
from plantvlm.io import read_json
from plantvlm.models import get_adapter
from plantvlm.training import Run, Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune a VLM")
    add_common_args(parser)
    parser.add_argument("--resume", default=None,
                        help="checkpoint directory to resume the adapter from")
    parser.add_argument("--tag", default="", help="suffix for the run directory")
    args = parser.parse_args()

    cfg = load(args)

    try:
        train_rows = load_split(cfg, "train")
        val_rows = load_split(cfg, "val")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    banner("Stage 7 — LoRA fine-tuning", cfg, (
        ("model", cfg.model.name),
        ("adapter", cfg.model.adapter),
        ("train", f"{len(train_rows)} rows / "
                  f"{len({r['image_id'] for r in train_rows})} images"),
        ("val", f"{len(val_rows)} rows / "
                f"{len({r['image_id'] for r in val_rows})} images"),
        ("batch", f"{cfg.train.micro_batch} x {cfg.train.grad_accum} = "
                  f"{cfg.effective_batch}"),
        ("lora", f"r={cfg.lora.r} alpha={cfg.lora.alpha} "
                 f"(vision tower excluded)"),
        ("augment", str(cfg.train.augment)),
    ))

    run = Run.create(cfg, tag=args.tag)
    run.write_lineage(cfg, read_json(cfg.paths.manifest))
    print(f"\n[run] {run.directory}")
    print("      config, data, and lineage frozen")

    print("\n[1/3] loading model and attaching LoRA...")
    adapter = get_adapter(cfg)
    trainer = Trainer(cfg, adapter, run, train_rows, val_rows,
                      resume_from=args.resume)

    print("[2/3] pre-flight checks...")
    pf = trainer.preflight()
    print(f"      supervised tokens : {pf['supervised_tokens']} "
          f"(sample uses {pf['tokens_used']}/{pf['max_length']})")
    print(f"      trainable params  : {pf['trainable_params']:,} "
          f"({pf['trainable_pct']:.3f}%)")
    print(f"      schedule          : {pf['steps_per_epoch']} steps/epoch, "
          f"{pf['total_steps']} total")
    if pf["truncating"]:
        print("      WARNING: samples reach max_length — captions may be truncated")

    print("\n[3/3] training\n")
    result = trainer.fit()

    print("=" * 74)
    print(f"  finished in {result.minutes:.1f} min"
          f"{' (early stopped)' if result.early_stopped else ''}")
    print(f"  best val loss {result.best_val_loss:.4f} at epoch {result.best_epoch}")
    print(f"  best adapter  -> {run.best}")
    print(f"  run directory -> {run.directory}")
    print("  Next: python scripts/03_infer.py --split val")
    print("=" * 74)


if __name__ == "__main__":
    main()

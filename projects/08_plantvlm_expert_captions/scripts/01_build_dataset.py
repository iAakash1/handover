"""
Stages 1-4 — Source -> pairs -> verification -> split -> JSONL.

    python scripts/01_build_dataset.py
    python scripts/01_build_dataset.py --set split.fold=2
    python scripts/01_build_dataset.py --verify-provenance

INPUT   the validated caption workbook + the image tree
OUTPUT  data/{pairs,train,val,test}.jsonl, data/manifest.json,
        data/caption_defects.json

Verification is a gate: if an invariant fails, nothing is written and the script
exits non-zero. A bad dataset must never reach training.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import add_common_args, banner, load

from plantvlm.data import build_dataset
from plantvlm.data.splits import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the training dataset")
    add_common_args(parser)
    parser.add_argument("--verify-provenance", action="store_true",
                        help="hash every image against the source dataset (slow)")
    args = parser.parse_args()

    cfg = load(args)
    banner("Stage 1-4 — build image/caption dataset", cfg, (
        ("source", cfg.paths.excel.name),
        ("images", str(cfg.paths.image_root)),
        ("output", str(cfg.paths.data_dir)),
    ))

    try:
        result = build_dataset(cfg, verify_provenance=args.verify_provenance)
    except Exception as exc:
        print(f"\nBUILD FAILED\n{exc}")
        sys.exit(1)

    manifest, defects = result["manifest"], result["defects"]

    print("\n[verification]")
    print(result["report"].render())

    print("\n[dataset]")
    print(f"  classes  : {len(manifest['classes'])}  {manifest['classes']}")
    print(f"  images   : {manifest['n_images']}")
    print(f"  pairs    : {manifest['n_pairs']}  "
          f"({manifest['captions_per_image']} captions x {manifest['n_images']} images)")
    lengths = [row["n_words"] for row in result["rows"]]
    print(f"  captions : {min(lengths)}-{max(lengths)} words "
          f"(mean {sum(lengths) / len(lengths):.1f})")

    print("\n[splits]  grouped by image, stratified by class, fold "
          f"{cfg.split.fold}")
    for name, info in summarize(result["splits"]).items():
        print(f"  {name:<6} {info['n_images']:>2} images  {info['n_rows']:>3} rows   "
              f"{info['images_per_class']}")

    if defects:
        print(f"\n[flagged]  {len(defects)} caption(s) need annotator attention "
              f"— reported, never edited")
        for d in defects:
            print(f"  {d.origin:<6} {d.disease:<24} {', '.join(d.flags)}")

    print(f"\n  written -> {cfg.paths.data_dir}")
    print("  Next: python scripts/00_verify.py")


if __name__ == "__main__":
    main()

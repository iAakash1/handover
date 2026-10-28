"""
dataset_builder.py
==================
Orchestrates the whole pipeline over a folder of images and writes every output
format you need.

For each image it:
    1. derives the ground-truth label (from sub-folder name or file name)
    2. looks the label up in the agronomy KB        -> Part A
    3. runs the visual analyzer on the pixels        -> Part B
    4. synthesises 5 captions fusing A + B
    5. records everything

Outputs written to config.OUTPUTS_DIR:
    captions_master.json   one rich object per image (A + B + 5 captions)  [for review]
    train.jsonl            one record per caption (20 imgs x 5 = 100 lines) [for training]
    captions_review.csv    flat sheet, easy to eyeball / hand-edit          [for your supervisor]
    visual_analysis_cache.json   caches Part B so re-runs skip the VLM
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import config
import agronomy_kb
import visual_analyzer
import caption_synthesizer


# --------------------------------------------------------------------------- #
# Discovering images + their labels
# --------------------------------------------------------------------------- #
def _iter_images(images_dir: str) -> List[Tuple[str, str]]:
    """
    Yield (image_path, raw_label) for every supported image under images_dir.

    Two layouts are supported:
      (a) sub-folders named after the class:
            images/Cotton___Bacterial_blight/img1.jpg   -> label = folder name
      (b) flat files whose name encodes the class:
            images/cotton___bacterial_blight_01.jpg     -> label = file stem
    """
    found: List[Tuple[str, str]] = []
    for root, _dirs, files in os.walk(images_dir):
        for fn in sorted(files):
            if not fn.lower().endswith(config.SUPPORTED_EXTS):
                continue
            path = os.path.join(root, fn)
            if os.path.abspath(root) != os.path.abspath(images_dir):
                # layout (a): label comes from the immediate parent folder
                label = os.path.basename(root)
            else:
                # layout (b): label comes from the file stem (strip a trailing _NN)
                stem = os.path.splitext(fn)[0]
                label = stem
            found.append((path, label))
    return found


# --------------------------------------------------------------------------- #
# Caching Part B (the expensive VLM step)
# --------------------------------------------------------------------------- #
def _load_cache() -> Dict:
    if os.path.exists(config.VISUAL_CACHE):
        try:
            with open(config.VISUAL_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict) -> None:
    with open(config.VISUAL_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(images_dir: Optional[str] = None,
          backend: Optional[str] = None,
          use_cache: bool = True) -> Dict:
    images_dir = images_dir or config.IMAGES_DIR
    backend = backend or config.VISUAL_BACKEND
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    pairs = _iter_images(images_dir)
    if not pairs:
        raise SystemExit(
            f"No images found under {images_dir}. Put your images there "
            f"(sub-folders named after the class, or files named like "
            f"cotton___bacterial_blight_01.jpg)."
        )

    cache = _load_cache() if use_cache else {}
    master: List[Dict] = []
    skipped: List[Tuple[str, str]] = []

    print(f"Found {len(pairs)} image(s). Visual backend: {backend}\n")

    for idx, (path, raw_label) in enumerate(pairs, 1):
        rel = os.path.relpath(path, config.PROJECT_ROOT)
        agro = agronomy_kb.parse_label(raw_label)
        if agro is None:
            print(f"[{idx:>2}/{len(pairs)}] SKIP  {rel}  (label '{raw_label}' not in KB)")
            skipped.append((rel, raw_label))
            continue

        cache_key = os.path.abspath(path)
        if use_cache and cache_key in cache:
            vis = cache[cache_key]
        else:
            vis = visual_analyzer.analyze_image(path, backend=backend)
            cache[cache_key] = vis
            if use_cache:
                _save_cache(cache)  # incremental save so a crash never loses work

        image_id = os.path.splitext(os.path.basename(path))[0]
        synth = caption_synthesizer.synthesize(agro, vis, image_id)

        master.append({
            "image_id": image_id,
            "image_path": rel.replace("\\", "/"),
            "raw_label": raw_label,
            "agronomy": {                      # Part A
                "crop": agro["crop"],
                "disease": agro["disease"],
                "pathogen": agro["pathogen"],
                "pathogen_type": agro["pathogen_type"],
                "canonical_symptoms": agro["symptoms"],
                "distinguishing": agro.get("distinguishing", ""),
                "management": agro.get("management", ""),
            },
            "visual": {                        # Part B
                "shot_type": vis["shot_type"],
                "lighting": vis["lighting"],
                "hand_or_object": vis["hand_or_object"],
                "background": vis["background"],
                "focus_quality": vis["focus_quality"],
                "leaf_wetness": vis["leaf_wetness"],
                "dominant_colors": vis["dominant_colors"],
                "observed_symptoms": vis["observed_symptoms"],
                "estimated_pct_affected": vis["estimated_pct_affected"],
                "image_quality_notes": vis["image_quality_notes"],
                "backend": vis.get("_backend", backend),
                "error": vis.get("_error"),
            },
            "severity": synth["severity"],
            "captions": synth["captions"],     # list of {archetype, text}
        })
        print(f"[{idx:>2}/{len(pairs)}] OK    {rel}  ->  {agro['crop']} / {agro['disease']} "
              f"({synth['severity']})")

    _write_master(master)
    _write_train_jsonl(master)
    _write_review_csv(master)

    print("\n" + "=" * 60)
    print(f"Images captioned : {len(master)}")
    print(f"Captions written : {len(master) * config.CAPTIONS_PER_IMAGE}")
    print(f"Skipped (no KB)  : {len(skipped)}")
    print(f"\nOutputs in {config.OUTPUTS_DIR}:")
    print(f"  - {os.path.basename(config.MASTER_JSON)}")
    print(f"  - {os.path.basename(config.TRAIN_JSONL)}")
    print(f"  - {os.path.basename(config.REVIEW_CSV)}")
    print("=" * 60)

    return {"master": master, "skipped": skipped}


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def _write_master(master: List[Dict]) -> None:
    with open(config.MASTER_JSON, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)


def _write_train_jsonl(master: List[Dict]) -> None:
    """
    Explode to one training record per caption, in a messages format ready for
    Qwen2.5-VL SFT. The <image> token marks where the image embedding goes.
    """
    with open(config.TRAIN_JSONL, "w", encoding="utf-8") as f:
        for rec in master:
            for cap in rec["captions"]:
                row = {
                    "image": rec["image_path"],
                    "messages": [
                        {"role": "user",
                         "content": f"<image>\n{config.INSTRUCTION}"},
                        {"role": "assistant", "content": cap["text"]},
                    ],
                    "meta": {
                        "image_id": rec["image_id"],
                        "crop": rec["agronomy"]["crop"],
                        "disease": rec["agronomy"]["disease"],
                        "pathogen": rec["agronomy"]["pathogen"],
                        "severity": rec["severity"],
                        "archetype": cap["archetype"],
                        "shot_type": rec["visual"]["shot_type"],
                        "lighting": rec["visual"]["lighting"],
                        "hand_or_object": rec["visual"]["hand_or_object"],
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_review_csv(master: List[Dict]) -> None:
    with open(config.REVIEW_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image_id", "image_path", "crop", "disease", "pathogen",
            "severity", "shot_type", "lighting", "hand_or_object", "background",
            "caption_index", "archetype", "caption",
        ])
        for rec in master:
            for i, cap in enumerate(rec["captions"], 1):
                w.writerow([
                    rec["image_id"], rec["image_path"],
                    rec["agronomy"]["crop"], rec["agronomy"]["disease"],
                    rec["agronomy"]["pathogen"], rec["severity"],
                    rec["visual"]["shot_type"], rec["visual"]["lighting"],
                    rec["visual"]["hand_or_object"], rec["visual"]["background"],
                    i, cap["archetype"], cap["text"],
                ])


if __name__ == "__main__":
    build()

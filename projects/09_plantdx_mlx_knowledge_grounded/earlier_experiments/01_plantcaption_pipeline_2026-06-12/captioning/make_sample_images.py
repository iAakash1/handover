"""
make_sample_images.py
=====================
Creates a small set of synthetic, class-labelled placeholder images so you can
run the whole pipeline end-to-end BEFORE you drop in your real 20 photos.

It writes sub-folders named exactly like real PlantVillage / field classes, each
with one or two generated images. Run once, then run the pipeline with the
"mock" backend to see the full output, or with "qwen"/"api" on real images.

    python make_sample_images.py
"""

from __future__ import annotations

import os
import random

from PIL import Image, ImageDraw

import config

# 20 class folders spanning the crops your supervisor named.
SAMPLE_CLASSES = [
    "Cotton___Bacterial_blight",
    "Cotton___Alternaria_leaf_spot",
    "Cotton___Grey_mildew",
    "Cotton___Leaf_curl_virus",
    "Sugarcane___Red_rot",
    "Sugarcane___Smut",
    "Sugarcane___Rust",
    "Sugarcane___Mosaic",
    "Maize___Common_rust",
    "Maize___Northern_leaf_blight",
    "Maize___Gray_leaf_spot",
    "Maize___Leaf_blight",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Yellow_leaf_curl_virus",
    "Tomato___healthy",
    "Potato___Late_blight",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
]


def _make_leaf_image(path: str, seed: int) -> None:
    rng = random.Random(seed)
    w = h = 320
    img = Image.new("RGB", (w, h), (235, 235, 230))
    d = ImageDraw.Draw(img)
    # a crude green leaf blob
    green = (rng.randint(40, 90), rng.randint(110, 160), rng.randint(40, 80))
    d.ellipse([40, 30, 280, 290], fill=green)
    d.line([160, 30, 160, 290], fill=(green[0] - 20, green[1] - 20, green[2] - 20), width=3)
    # scatter some "lesion" spots
    for _ in range(rng.randint(4, 14)):
        cx, cy = rng.randint(70, 250), rng.randint(60, 260)
        r = rng.randint(4, 16)
        spot = (rng.randint(90, 150), rng.randint(50, 90), rng.randint(20, 50))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=spot)
    img.save(path, quality=85)


def main() -> None:
    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    count = 0
    for i, cls in enumerate(SAMPLE_CLASSES):
        folder = os.path.join(config.IMAGES_DIR, cls)
        os.makedirs(folder, exist_ok=True)
        out = os.path.join(folder, f"{cls.replace(',', '').replace(' ', '_')}_01.jpg")
        _make_leaf_image(out, seed=1000 + i)
        count += 1
    print(f"Wrote {count} sample images under {config.IMAGES_DIR}")
    print("Now run:  python run_pipeline.py --backend mock")


if __name__ == "__main__":
    main()

"""
run_pipeline.py
===============
Command-line entry point for the PlantCaption pipeline.

Examples
--------
    # 0. (optional) make synthetic test images
    python make_sample_images.py

    # 1. dry run with no model at all (tests the plumbing in seconds)
    python run_pipeline.py --backend mock

    # 2. real run with your local Qwen2.5-VL on your 20 images
    python run_pipeline.py --backend qwen

    # re-run but ignore the cached visual analysis
    python run_pipeline.py --backend qwen --no-cache
"""

from __future__ import annotations

import argparse

import config
import dataset_builder


def main() -> None:
    p = argparse.ArgumentParser(description="Generate 5 captions per plant-disease image.")
    p.add_argument("--images", default=config.IMAGES_DIR,
                   help="folder containing the images (default: %(default)s)")
    p.add_argument("--backend", default=config.VISUAL_BACKEND,
                   choices=["qwen", "mock"],
                   help="visual-analysis backend (default: %(default)s)")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore and overwrite the cached visual analysis")
    args = p.parse_args()

    dataset_builder.build(
        images_dir=args.images,
        backend=args.backend,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()

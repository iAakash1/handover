"""
config.py
=========
Central configuration for the PlantCaption pipeline.

Everything that you might want to tune lives here so the rest of the code
never hard-codes a path or a magic number. Edit this file, not the others.
"""

from __future__ import annotations

import os
from typing import Dict

# --------------------------------------------------------------------------- #
# 1. PATHS
# --------------------------------------------------------------------------- #
# Project root = the folder that CONTAINS the `captioning/` package.
# On your machine this will resolve to something like C:/Users/Admin/Desktop/VLM/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where your 20 demo images live. You can organise them in EITHER way:
#   (a) flat folder of images named like  cotton___bacterial_blight_01.jpg
#   (b) sub-folders named after the class:  images/Cotton___Bacterial_blight/xxx.jpg
# The label parser in agronomy_kb.py understands both.
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")

# All generated artefacts land here.
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

MASTER_JSON = os.path.join(OUTPUTS_DIR, "captions_master.json")   # rich, per-image
TRAIN_JSONL = os.path.join(OUTPUTS_DIR, "train.jsonl")            # exploded, training-ready
REVIEW_CSV = os.path.join(OUTPUTS_DIR, "captions_review.csv")     # human-friendly sheet
VISUAL_CACHE = os.path.join(OUTPUTS_DIR, "visual_analysis_cache.json")  # avoids re-running VLM

# --------------------------------------------------------------------------- #
# 2. VISUAL-ANALYSIS BACKEND  (Part B of the model: "what does the image look like?")
# --------------------------------------------------------------------------- #
# "qwen"   -> local Qwen2.5-VL-7B-Instruct  (default; you already run this)
# "mock"   -> no model at all; deterministic fake analysis for plumbing tests
VISUAL_BACKEND = os.environ.get("PLANTCAP_BACKEND", "qwen")

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_MAX_NEW_TOKENS = 512
QWEN_DTYPE = "bfloat16"            # matches your training setup
QWEN_MAX_IMAGE_SIDE = 1024        # downscale long edge before inference to save VRAM

# --------------------------------------------------------------------------- #
# 3. CAPTION GENERATION  (Part A x Part B fused)
# --------------------------------------------------------------------------- #
CAPTIONS_PER_IMAGE = 5            # COCO standard; your supervisor asked for 5

# Severity is derived from the % of leaf area the visual model says is affected.
# (lower bound inclusive, upper bound exclusive)
SEVERITY_BANDS: Dict[str, tuple] = {
    "healthy":  (0, 3),
    "mild":     (3, 15),
    "moderate": (15, 45),
    "severe":   (45, 101),
}

# A fixed instruction string so that every training record asks the model the
# same question. Keep this identical to the one you fine-tune with.
INSTRUCTION = (
    "You are an expert plant pathologist. Examine this image and describe both "
    "what you visually observe in the photograph and the plant-health condition "
    "it shows, including the crop, the disease, its severity, the causal agent, "
    "and the key visible symptoms."
)

# --------------------------------------------------------------------------- #
# 4. MISC
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

"""
config.py — single source of truth for the Crop Disease VLM Benchmark Suite.

Every other script imports from here instead of hardcoding paths, model IDs,
the evaluation prompt, or rubric weights. If you change something (add a
model, change the prompt, re-weight the rubric), change it here once.

All paths are resolved relative to this file's location, so the project
can be moved, renamed, or zipped/unzipped anywhere without breaking.
"""

import os
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent
IMAGES_DIR       = PROJECT_ROOT / "images"
RESULTS_DIR      = PROJECT_ROOT / "results"
JSON_RESULTS_DIR = PROJECT_ROOT / "json_results"
CSV_DIR          = PROJECT_ROOT / "csv"
PLOTS_DIR        = PROJECT_ROOT / "plots"
REPORTS_DIR      = PROJECT_ROOT / "reports"
SCRIPTS_DIR      = PROJECT_ROOT / "scripts"

# ── Models — MLX-VLM, Apple Silicon, 4-bit quantized ───────────────────────
MODELS = {
    "qwen25vl":      "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "gemma3":        "mlx-community/gemma-3-12b-it-4bit",
    "internvl3":     "mlx-community/InternVL3-8B-MLX-4bit",
    "llama32vision": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
}

# ── Image discovery ──────────────────────────────────────────────────────
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
                     ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}

# ── Inference prompt — identical for every model, every image ─────────────
PROMPT = (
    "You are an agricultural expert.\n\n"
    "Analyze this crop image carefully.\n\n"
    "Return:\n\n"
    "Plant species:\n"
    "Visible symptoms:\n"
    "Leaf color:\n"
    "Leaf texture:\n"
    "Spots or lesions:\n"
    "Severity:\n"
    "Most likely disease:\n"
    "Alternative diagnoses:\n"
    "Confidence (0-100):\n"
    "Reasoning:\n\n"
    "Only describe what is visible.\n"
    "If uncertain, explicitly state uncertainty.\n"
    "Do not hallucinate diseases."
)

MAX_TOKENS  = 600
TEMPERATURE = 0.0
RANDOM_SEED = 42

# ── LLM-as-a-judge settings ─────────────────────────────────────────────────
# API keys are read from environment variables ONLY. Never hardcode keys here.
#   export OPENAI_API_KEY="sk-..."          (provider = openai, default)
#   export GEMINI_API_KEY="..."             (provider = gemini)
#   export VLM_JUDGE_PROVIDER="gemini"      (to switch providers)
JUDGE_PROVIDER     = os.environ.get("VLM_JUDGE_PROVIDER", "openai").lower()  # "openai" | "gemini"
OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
OPENAI_JUDGE_MODEL  = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4.1")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
GEMINI_JUDGE_MODEL  = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-2.5-pro")
JUDGE_MAX_RETRIES  = 3
JUDGE_RETRY_DELAY  = 5  # seconds between retries

# Rubric weights — MUST sum to 100. "hallucination" is scored as a
# groundedness score: 20 = fully grounded, 0 = severe fabrication
# (i.e. HIGHER is better, despite the name "hallucination").
RUBRIC_WEIGHTS = {
    "plant_species":          10,
    "symptom_quality":        20,
    "disease_quality":        40,
    "hallucination":          20,
    "confidence_calibration": 10,
}
assert sum(RUBRIC_WEIGHTS.values()) == 100, "RUBRIC_WEIGHTS must sum to 100"

RUBRIC_LABELS = {
    "plant_species":          "Plant species correctness",
    "symptom_quality":        "Symptom description quality",
    "disease_quality":        "Disease prediction quality",
    "hallucination":          "Hallucination / groundedness (higher = more grounded)",
    "confidence_calibration": "Confidence calibration",
}


def ensure_directories():
    """Create every directory the pipeline writes to, if missing."""
    for d in [IMAGES_DIR, RESULTS_DIR, JSON_RESULTS_DIR, CSV_DIR, PLOTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for model_key in MODELS:
        (RESULTS_DIR / model_key).mkdir(parents=True, exist_ok=True)

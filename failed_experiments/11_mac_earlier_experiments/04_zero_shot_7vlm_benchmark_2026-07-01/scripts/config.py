"""
Central configuration for the PlantDx zero-shot VLM benchmark.

Everything that might reasonably need tweaking lives here: which models
run, the exact prompt sent to them, where the dataset/outputs/results
live, and generation settings. No other file should need editing for
routine changes (e.g. adding a model, pointing at a different dataset).
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
# Resolved relative to this file, so the project runs correctly no matter
# what directory you invoke `python scripts/run_benchmark.py` from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# ── Models ───────────────────────────────────────────────────────────────
# Every model is loaded and run through the identical code path in
# scripts/model_runner.py. Add, remove, or reorder freely.
MODELS = [
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    "mlx-community/gemma-3-12b-it-4bit",
    "mlx-community/InternVL3-8B-MLX-4bit",
    "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
    "mlx-community/Molmo-7B-D-0924-4bit",
    "mlx-community/deepseek-vl2-small-4bit",
]

# ── Prompt ───────────────────────────────────────────────────────────────
# Sent VERBATIM and IDENTICALLY to every model, for every image. Do not
# customize this per model or per class — that would invalidate the
# zero-shot comparison. Folder names are never substituted in here.
PROMPT = """You are an expert agricultural plant pathologist.
This image is of a TOMATO crop.
Analyze the leaf carefully.
Return ONLY the following fields.
Crop:
Visible Symptoms:
Disease:
Alternative Diagnoses:
Severity:
Confidence (0-100):
Reasoning:
Do not use external knowledge.
Only describe what can be visually observed.
If uncertain, explicitly say so instead of guessing."""

# ── Generation settings ─────────────────────────────────────────────────
MAX_TOKENS = 512  # generous headroom so the Reasoning field isn't truncated
TEMPERATURE = 0.0  # deterministic, reproducible outputs across runs
# Applied to mx.random at startup as a defensive extra layer of
# determinism, on top of TEMPERATURE=0.0 (already deterministic on its
# own for greedy decoding).
SEED = 42

# ── Dataset ──────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

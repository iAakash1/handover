"""
scripts/utils.py — shared helpers used by every pipeline stage.

Notable contents:
  - list_images / load_json / save_json / log     : plumbing
  - parse_model_output                              : turns the fixed-field
        agricultural-expert response into a dict of fields
  - normalize_label / labels_match                  : fuzzy disease-name
        comparison so "Late blight (Phytophthora infestans)" matches
        ground truth "Late Blight"
  - run_mlx_vlm_generate                             : a defensive wrapper
        around mlx_vlm.generate(). mlx-vlm's public API (helper names,
        kwarg names, return type) has changed across releases. This
        wrapper tries the documented current call shape first, falls
        back to an older shape on TypeError, and recovers latency /
        tokens-per-sec / peak-memory stats from mlx-vlm's own verbose
        stdout output when possible — falling back to manual timing if
        that format ever changes. If generate() raises in a way this
        wrapper doesn't anticipate, see the README troubleshooting
        section for how to adapt it to your installed mlx-vlm version.
"""

import sys
import io
import re
import json
import time
import difflib
import contextlib
from pathlib import Path

# Make the project root importable regardless of how/where this module
# is imported from.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config  # noqa: E402


# ── Logging ──────────────────────────────────────────────────────────────
def log(msg: str):
    print(msg, flush=True)


# ── Filesystem / JSON plumbing ───────────────────────────────────────────
def list_images(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix in config.VALID_IMAGE_EXTS
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ── Parsing the fixed-field model response ───────────────────────────────
FIELD_LABELS = [
    "Plant species",
    "Visible symptoms",
    "Leaf color",
    "Leaf texture",
    "Spots or lesions",
    "Severity",
    "Most likely disease",
    "Alternative diagnoses",
    "Confidence (0-100)",
    "Reasoning",
]

FIELD_KEYS = {
    "Plant species":          "plant_species",
    "Visible symptoms":       "visible_symptoms",
    "Leaf color":             "leaf_color",
    "Leaf texture":           "leaf_texture",
    "Spots or lesions":       "spots_or_lesions",
    "Severity":               "severity",
    "Most likely disease":    "most_likely_disease",
    "Alternative diagnoses":  "alternative_diagnoses",
    "Confidence (0-100)":     "confidence_raw",
    "Reasoning":              "reasoning",
}

_FIELD_PATTERN = re.compile(
    "(" + "|".join(re.escape(l) for l in FIELD_LABELS) + r")\s*:?\s*",
    re.IGNORECASE,
)


def parse_model_output(raw_text: str) -> dict:
    """
    Parse the fixed-field agricultural-expert prompt output into a dict.
    Robust to minor formatting drift (markdown bold, extra colons, fields
    appearing out of order). Any field the model didn't produce comes
    back as an empty string; 'confidence' is a float or None.
    """
    result = {v: "" for v in FIELD_KEYS.values()}
    result["confidence"] = None
    if not raw_text:
        return result

    text = raw_text.replace("**", "").replace("*", "")  # strip markdown emphasis

    matches = list(_FIELD_PATTERN.finditer(text))
    for i, m in enumerate(matches):
        label = m.group(1)
        canon = next((l for l in FIELD_LABELS if l.lower() == label.lower()), None)
        if canon is None:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[start:end].strip().strip("-").strip()
        result[FIELD_KEYS[canon]] = value

    conf_match = re.search(r"\d+(\.\d+)?", result.get("confidence_raw", "") or "")
    result["confidence"] = float(conf_match.group()) if conf_match else None
    return result


# ── Fuzzy disease-label comparison ───────────────────────────────────────
def normalize_label(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)            # drop parenthetical scientific names
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def labels_match(predicted: str, ground_truth: str, threshold: float = 0.6) -> bool:
    """
    True if `predicted` plausibly refers to the same disease as
    `ground_truth`. Used both for accuracy scoring and for canonicalizing
    free-text predictions onto the known ground-truth label set for
    confusion-matrix plotting.
    """
    if not predicted or not ground_truth or ground_truth.lower() == "unknown":
        return False
    p, g = normalize_label(predicted), normalize_label(ground_truth)
    if not p or not g:
        return False
    if g in p or p in g:
        return True
    ratio = difflib.SequenceMatcher(None, p, g).ratio()
    return ratio >= threshold


# ── Apple-silicon / mlx memory introspection (best effort) ───────────────
def reset_peak_memory():
    try:
        import mlx.core as mx
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()
    except Exception:
        pass


def get_peak_memory_gb():
    """
    Best-effort peak unified-memory usage in GB. mlx's memory
    introspection API has moved between `mx.metal.get_peak_memory()` and
    `mx.get_peak_memory()` across releases, so we try both and fall back
    to the current process RSS via psutil (a real but noisier proxy)
    if neither is available.
    """
    try:
        import mlx.core as mx
        if hasattr(mx, "get_peak_memory"):
            return mx.get_peak_memory() / 1e9
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            return mx.metal.get_peak_memory() / 1e9
    except Exception:
        pass
    try:
        import psutil
        import os as _os
        proc = psutil.Process(_os.getpid())
        return proc.memory_info().rss / 1e9
    except Exception:
        return None


# ── Defensive mlx-vlm generate() wrapper ──────────────────────────────────
def run_mlx_vlm_generate(model, processor, model_config, prompt_text, image_path,
                          max_tokens=600, temperature=0.0):
    """
    Run one (model, image) inference and return:
      {text, latency_sec, tokens_per_sec, peak_memory_gb, raw_stdout, error}

    A single incompatible mlx-vlm version should not crash the whole
    benchmark — every failure mode here is caught and reported in
    `error` rather than raised, except the case where mlx_vlm itself
    cannot be imported at all (handled one level up, in benchmark.py).
    """
    from mlx_vlm import generate as mlx_generate
    try:
        from mlx_vlm.prompt_utils import apply_chat_template
    except Exception:
        apply_chat_template = None

    formatted_prompt = prompt_text
    if apply_chat_template is not None and model_config is not None:
        try:
            formatted_prompt = apply_chat_template(
                processor, model_config, prompt_text, num_images=1
            )
        except Exception:
            formatted_prompt = prompt_text

    reset_peak_memory()
    stdout_buffer = io.StringIO()
    error = None
    raw_return = None
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            try:
                raw_return = mlx_generate(
                    model, processor, formatted_prompt,
                    image=[str(image_path)],
                    max_tokens=max_tokens, temp=temperature, verbose=True,
                )
            except TypeError:
                # Older/newer mlx-vlm signature: positional image path,
                # "temperature" instead of "temp".
                raw_return = mlx_generate(
                    model, processor, formatted_prompt, str(image_path),
                    max_tokens=max_tokens, temperature=temperature, verbose=True,
                )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    captured = stdout_buffer.getvalue()

    # Normalize the returned value to plain text.
    text = ""
    if isinstance(raw_return, str):
        text = raw_return
    elif raw_return is not None:
        text = getattr(raw_return, "text", None) or str(raw_return)

    # mlx-vlm's verbose mode prints lines like:
    #   "Generation: 64 tokens, 12.345 tokens-per-sec"
    #   "Peak memory: 8.234 GB"
    # Recover these if present; otherwise compute our own estimate.
    gen_tps = None
    peak_mem_gb = None
    m = re.search(r"Generation:.*?([\d.]+)\s*tokens-per-sec", captured)
    if m:
        gen_tps = float(m.group(1))
    m = re.search(r"Peak memory:\s*([\d.]+)\s*GB", captured)
    if m:
        peak_mem_gb = float(m.group(1))

    if gen_tps is None and text and elapsed > 0:
        try:
            n_tokens = len(processor.tokenizer.encode(text))
        except Exception:
            n_tokens = len(text.split())
        gen_tps = (n_tokens / elapsed) if elapsed > 0 else None

    if peak_mem_gb is None:
        peak_mem_gb = get_peak_memory_gb()

    return {
        "text": text.strip(),
        "latency_sec": elapsed,
        "tokens_per_sec": gen_tps,
        "peak_memory_gb": peak_mem_gb,
        "raw_stdout": captured.strip()[-2000:],
        "error": error,
    }

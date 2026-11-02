"""
visual_analyzer.py
==================
Part B of the model: the IMAGE-PATTERN UNDERSTANDING.

This module looks at the actual photograph and returns a structured description
of the *scene* -- the things a label can never tell you:

    * is a hand or finger or ruler holding the leaf?
    * is the shot a tight close-up or a wide field view?
    * is the lighting bright daylight, overcast, indoor, flash, or dark?
    * what is the background (soil, foliage, sky, plain studio, blurred)?
    * is the image sharp or blurred?
    * what symptoms are actually visible, and roughly what % of the leaf is affected?

These are exactly the cues your supervisor asked you to capture
("if there is a hand involved, is the image in dark condition?"). Together with
the agronomy KB (Part A) they let the caption describe BOTH the agriculture and
the image itself.

Two backends are supported (set in config.VISUAL_BACKEND):
    "qwen"  -> local Qwen2.5-VL-7B-Instruct  (default)
    "mock"  -> deterministic fake output, no model, for testing the plumbing
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional

from PIL import Image

import config


# --------------------------------------------------------------------------- #
# The strict-JSON prompt that drives whichever vision model we use.
# Keeping it identical across backends makes their outputs interchangeable.
# --------------------------------------------------------------------------- #
VISION_PROMPT = (
    "You are an expert agricultural image analyst. Look ONLY at this photograph "
    "and report what is physically visible. Do NOT guess the disease name. "
    "Return a single JSON object and nothing else, with exactly these keys:\n"
    '  "shot_type": one of '
    "[extreme_closeup, single_leaf_closeup, multiple_leaves, whole_plant, field_wide]\n"
    '  "lighting": one of '
    "[bright_daylight, diffuse_overcast, indoor_artificial, camera_flash, low_light_dark, backlit]\n"
    '  "hand_or_object": short phrase describing any human hand, fingers, ruler, '
    "tool, pot or holder visible, or \"none\"\n"
    '  "background": one of '
    "[soil_or_ground, dense_foliage, sky, plain_or_studio, blurred_natural, indoor_surface, other]\n"
    '  "focus_quality": one of [sharp, slightly_soft, blurred]\n'
    '  "leaf_wetness": one of [dry, dew_or_droplets, wet_glossy, unclear]\n'
    '  "dominant_colors": short list of colour words\n'
    '  "observed_symptoms": one sentence describing visible spots, lesions, '
    "discolouration or growth -- their colour, shape and where they sit\n"
    '  "estimated_pct_affected": integer 0-100, approximate share of visible '
    "leaf area showing any symptom\n"
    '  "image_quality_notes": short note on glare, shadow, blur, occlusion or '
    "multiple subjects, or \"none\"\n"
    "Respond with raw JSON only, no markdown fences, no commentary."
)

# Fields we promise to return no matter what, so downstream code is safe.
_REQUIRED_KEYS = {
    "shot_type", "lighting", "hand_or_object", "background", "focus_quality",
    "leaf_wetness", "dominant_colors", "observed_symptoms",
    "estimated_pct_affected", "image_quality_notes",
}

_FALLBACK = {
    "shot_type": "single_leaf_closeup",
    "lighting": "bright_daylight",
    "hand_or_object": "none",
    "background": "plain_or_studio",
    "focus_quality": "sharp",
    "leaf_wetness": "dry",
    "dominant_colors": ["green"],
    "observed_symptoms": "visible discolouration on the leaf surface",
    "estimated_pct_affected": 20,
    "image_quality_notes": "none",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    # Downscale long edge to save VRAM / tokens without losing symptom detail.
    long_edge = max(img.size)
    if long_edge > config.QWEN_MAX_IMAGE_SIDE:
        scale = config.QWEN_MAX_IMAGE_SIDE / long_edge
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def _extract_json(text: str) -> Dict:
    """Pull the first {...} block out of a model response and parse it."""
    text = text.strip()
    # Strip accidental markdown fences.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start:end + 1])


def _coerce(raw: Dict) -> Dict:
    """Fill any missing keys from the fallback and sanitise types."""
    out = dict(_FALLBACK)
    for k in _REQUIRED_KEYS:
        if k in raw and raw[k] not in (None, ""):
            out[k] = raw[k]
    # Make sure the percentage is an int in range.
    try:
        pct = int(round(float(out["estimated_pct_affected"])))
    except (ValueError, TypeError):
        pct = _FALLBACK["estimated_pct_affected"]
    out["estimated_pct_affected"] = max(0, min(100, pct))
    # dominant_colors must be a list.
    if isinstance(out["dominant_colors"], str):
        out["dominant_colors"] = [c.strip() for c in re.split(r"[,/]", out["dominant_colors"]) if c.strip()]
    return out


# --------------------------------------------------------------------------- #
# Backend: MOCK  (no model, deterministic; lets you test everything offline)
# --------------------------------------------------------------------------- #
def _analyze_mock(path: str) -> Dict:
    # Derive pseudo-random-but-stable variety from the filename so different
    # images get different scenes during a plumbing test.
    h = abs(hash(os.path.basename(path)))
    shots = ["extreme_closeup", "single_leaf_closeup", "multiple_leaves", "whole_plant", "field_wide"]
    lights = ["bright_daylight", "diffuse_overcast", "indoor_artificial", "low_light_dark", "backlit"]
    hands = ["none", "a hand holding the leaf by the petiole", "two fingers pinning the leaf flat", "none"]
    bgs = ["soil_or_ground", "dense_foliage", "plain_or_studio", "blurred_natural"]
    out = dict(_FALLBACK)
    out.update({
        "shot_type": shots[h % len(shots)],
        "lighting": lights[h % len(lights)],
        "hand_or_object": hands[h % len(hands)],
        "background": bgs[h % len(bgs)],
        "estimated_pct_affected": 5 + (h % 60),
        "observed_symptoms": "scattered discoloured patches across the blade",
    })
    return out


# --------------------------------------------------------------------------- #
# Backend: QWEN2.5-VL  (local, default)
# --------------------------------------------------------------------------- #
_QWEN = {"model": None, "processor": None}


def _load_qwen():
    if _QWEN["model"] is not None:
        return
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _QwenVL
    except ImportError:  # older transformers naming
        from transformers import Qwen2VLForConditionalGeneration as _QwenVL

    dtype = torch.bfloat16 if config.QWEN_DTYPE == "bfloat16" else torch.float16
    print(f"[visual] loading {config.QWEN_MODEL_ID} ({config.QWEN_DTYPE}) ...")
    _QWEN["processor"] = AutoProcessor.from_pretrained(config.QWEN_MODEL_ID)
    _QWEN["model"] = _QwenVL.from_pretrained(
        config.QWEN_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    ).eval()
    print("[visual] Qwen2.5-VL ready.")


def _analyze_qwen(path: str) -> Dict:
    import torch
    _load_qwen()
    model, processor = _QWEN["model"], _QWEN["processor"]
    image = _load_image(path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": VISION_PROMPT},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # NOTE: image_grid_thw is produced automatically by the processor here and
    # carried in `inputs`; we never strip it. (This was the bug class you hit
    # during training -- in inference the processor handles it for us.)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=config.QWEN_MAX_NEW_TOKENS,
            do_sample=False,
        )
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    reply = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return _coerce(_extract_json(reply))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
_DISPATCH = {"qwen": _analyze_qwen, "mock": _analyze_mock}


def analyze_image(path: str, backend: Optional[str] = None) -> Dict:
    """
    Return a sanitised visual-scene dict for one image. Never raises on a model
    hiccup -- it falls back to a safe default and records the error -- so a
    single bad image cannot stop a 20-image batch.
    """
    backend = backend or config.VISUAL_BACKEND
    fn = _DISPATCH.get(backend)
    if fn is None:
        raise ValueError(f"unknown VISUAL_BACKEND '{backend}'")
    try:
        result = fn(path)
        result["_backend"] = backend
        result["_error"] = None
        return result
    except Exception as exc:  # noqa: BLE001 - we want any failure to be soft
        result = dict(_FALLBACK)
        result["_backend"] = backend
        result["_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[visual] WARNING on {os.path.basename(path)}: {result['_error']}")
        return result


if __name__ == "__main__":
    # Smoke-test the mock backend on a throwaway image.
    test_path = "/tmp/_vt.jpg"
    Image.new("RGB", (256, 256), (60, 120, 60)).save(test_path)
    print(json.dumps(analyze_image(test_path, backend="mock"), indent=2))

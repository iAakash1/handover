"""Inference for the demo — a thin, honest reuse of the frozen PlantDx code.

Model loading, adapter handling, and prompt templating are reused from
``plantdx.training.inference`` / ``plantdx.evaluation``. This module adds only
what a *demo* needs on top and that the reused code doesn't expose:

* a **real** generation confidence (mean probability of the model's own selected
  tokens), computed by iterating ``mlx_vlm.stream_generate`` — the one-shot
  ``generate`` only returns the last token's full-vocab logprobs, which is not a
  usable confidence;
* per-run telemetry (latency, tokens, peak memory);
* adapter verification (rank, target modules, trainable-param count) so the UI
  can prove the LoRA adapter — not the base model — is attached;
* robust, crop-scoped disease classification (see ``app.classification``).

Nothing here changes training, evaluation, or their recorded numbers.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st

from app.classification import UNCLASSIFIED, classify
from app.logging_setup import get_logger
from app.utils import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DKB_PATH,
    MODEL_ID,
    crop_profile,
    format_seconds,
    pretty_disease,
    strip_crop_suffix,
)
from plantdx.core.exceptions import PlantDxError
from plantdx.evaluation.checkpoint import adapter_weights_path, resolve_adapter_dir
from plantdx.training.inference import LoadedModel, load_model
from plantdx.utils.hashing import sha256_bytes

_MAX_TOKENS = 128
_TEMPERATURE = 0.0  # deterministic (greedy) decoding — matches evaluation
_ADAPTER_CONFIG = "adapter_config.json"

# Prediction status buckets (drive the diagnosis text + UI treatment).
STATUS_CONFIDENT = "confident"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_UNKNOWN = "unknown"

_log = get_logger()


# --------------------------------------------------------------------------- #
# Disease Knowledge Base lookup (read-only; the demo never edits the DKB)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _dkb_by_id() -> dict[str, dict[str, Any]]:
    if not DKB_PATH.is_file():
        return {}
    data = json.loads(DKB_PATH.read_text(encoding="utf-8"))
    return {str(d["id"]): d for d in data.get("diseases", [])}


def _disease_info(disease_id: str) -> dict[str, Any]:
    """Grounded display fields for a predicted disease id, straight from the DKB."""
    entry = _dkb_by_id().get(disease_id, {})
    symptoms = list(entry.get("primary_symptoms") or [])
    features = list(entry.get("diagnostic_visual_features") or [])
    shown = features or symptoms  # prefer concise diagnostic features
    disease_name = strip_crop_suffix(str(entry.get("disease", ""))) or pretty_disease(disease_id)
    return {
        "disease_name": disease_name,
        "common_name": str(entry.get("common_name", "")) or None,
        "symptoms": shown[:8],
    }


# --------------------------------------------------------------------------- #
# Cached model loading — one load per (model, adapter), reused across runs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelHandle:
    """A loaded model plus how long it took to load (measured once)."""

    loaded: LoadedModel
    adapter_dir: str
    load_seconds: float


@st.cache_resource(show_spinner=False, max_entries=1)
def get_model_handle(model_path: str, adapter_dir: str) -> ModelHandle:
    """Load (and cache) the base model + adapter. Reused for every prediction.

    Cached by Streamlit on ``(model_path, adapter_dir)``. ``max_entries=1`` keeps
    only one crop's 7B model resident — switching crops evicts the previous one
    instead of holding two 7B models in 24 GB of unified memory. Because the
    cache persists across reruns and sessions, a prediction never reloads it.
    """
    _configure_mlx_memory()
    start = time.perf_counter()
    loaded = load_model(model_path, adapter_path=adapter_dir)
    elapsed = time.perf_counter() - start
    _log.info("model loaded model=%s adapter=%s in %.1fs", model_path, adapter_dir, elapsed)
    return ModelHandle(loaded=loaded, adapter_dir=adapter_dir, load_seconds=elapsed)


def _configure_mlx_memory() -> None:
    """Bound MLX's Metal buffer cache so repeated inferences don't exhaust RAM."""
    try:
        import mlx.core as mx

        # Cap the reusable buffer cache (not the working set) at 4 GB.
        mx.set_cache_limit(4 * 1024**3)
    except Exception:
        pass


def _release_mlx_cache() -> None:
    """Best-effort release of MLX's Metal buffer cache (never fatal)."""
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _mlx_active_memory_gb() -> float | None:
    try:
        import mlx.core as mx

        return float(mx.get_active_memory()) / 1e9
    except Exception:
        return None


def resolve_crop_adapter(crop: str) -> Path:
    """Validate and return the adapter directory for ``crop`` (friendly errors)."""
    profile = crop_profile(crop)
    if not profile.adapter_dir.exists():
        raise PlantDxError(
            f"No trained {profile.label} adapter found at "
            f"`{profile.adapter_dir}`. Train it first with "
            f"`plantdx train --config configs/train/qwen25vl_{profile.crop}.yaml`, "
            "or select the other crop."
        )
    # resolve_adapter_dir raises a clear DerivationError if a required file is missing.
    return resolve_adapter_dir(str(profile.adapter_dir))


def load_for_crop(crop: str) -> ModelHandle:
    """Resolve the crop's adapter and return the cached model handle."""
    adapter_dir = resolve_crop_adapter(crop)
    return get_model_handle(MODEL_ID, str(adapter_dir))


# --------------------------------------------------------------------------- #
# Adapter verification — prove the LoRA adapter (not the base model) is attached
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=4)
def adapter_info(adapter_dir: str) -> dict[str, Any]:
    """Read adapter config + weights and summarize what's attached.

    Returns fine-tune type, LoRA rank/scale, number of adapted modules, total
    trainable-parameter count, weights checksum, and ``attached=True`` — enough
    for the UI to confirm the trained LoRA adapter is in use, not the base model.
    """
    d = Path(adapter_dir)
    cfg_path = d / _ADAPTER_CONFIG
    info: dict[str, Any] = {
        "adapter_dir": str(d),
        "fine_tune_type": None,
        "rank": None,
        "scale": None,
        "num_target_modules": None,
        "trainable_params": None,
        "lora_tensor_count": None,
        "weights_checksum": None,
        "attached": False,
    }
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        lora = cfg.get("lora_parameters", {})
        keys = lora.get("keys", [])
        info.update(
            fine_tune_type=cfg.get("fine_tune_type"),
            rank=lora.get("rank"),
            scale=lora.get("scale"),
            num_target_modules=len(keys) if isinstance(keys, list) else None,
        )
    weights = adapter_weights_path(d)
    if weights.is_file():
        params, tensors = _count_safetensors_params(weights)
        info.update(
            trainable_params=params,
            lora_tensor_count=tensors,
            weights_checksum=f"sha256:{sha256_bytes(weights.read_bytes())[:16]}",
            attached=bool(tensors),
        )
    return info


def _count_safetensors_params(path: Path) -> tuple[int, int]:
    """Total element count and tensor count in a safetensors file (header-only)."""
    try:
        from safetensors import safe_open

        total, count = 0, 0
        with safe_open(str(path), framework="numpy") as f:  # type: ignore[no-untyped-call]
            for key in f.keys():  # noqa: SIM118 - safetensors handle isn't a dict
                shape = f.get_slice(key).get_shape()
                n = 1
                for dim in shape:
                    n *= int(dim)
                total += n
                count += 1
        return total, count
    except Exception:
        return 0, 0


def mlx_import_status() -> tuple[bool, str]:
    """Whether ``mlx-vlm`` imports here, and — if not — why.

    Returns ``(ok, reason)`` where ``reason`` is one of ``""`` (ok),
    ``"missing"`` (not installed), or ``"numpy_abi"`` (installed, but its
    transformers/numba dependency chain hits a NumPy 1.x-vs-2.x ABI conflict —
    the documented environment issue, not an app or model problem).
    """
    try:
        import mlx_vlm  # noqa: F401

        return True, ""
    except (KeyboardInterrupt, SystemExit):  # never swallow these
        raise
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("mlx"):
            return False, "missing"
        return False, "numpy_abi" if _looks_like_numpy_abi(str(exc)) else "missing"
    except BaseException as exc:
        # ImportError/AttributeError (and, from broken numba/NumPy C extensions,
        # occasionally lower-level errors) all mean "the backend won't import here".
        return False, "numpy_abi" if _looks_like_numpy_abi(str(exc)) else "other"


def _looks_like_numpy_abi(message: str) -> bool:
    needles = ("numpy.core.multiarray", "numba", "_ARRAY_API", "NumPy 1.x", "numpy 2")
    low = message.lower()
    return any(n.lower() in low for n in needles)


# --------------------------------------------------------------------------- #
# One prediction
# --------------------------------------------------------------------------- #


def run_inference(
    handle: ModelHandle,
    image_path: str | Path,
    crop: str,
    *,
    max_tokens: int = _MAX_TOKENS,
    temperature: float = _TEMPERATURE,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Caption one image, classify it, and score generation confidence.

    Uses ``stream_generate`` so the mean probability of the model's own selected
    tokens can be measured — a real confidence, unlike the one-shot ``generate``
    whose only logprobs are the last token's full-vocab distribution.
    """
    from mlx_vlm import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    loaded = handle.loaded
    profile = crop_profile(crop)
    prompt = apply_chat_template(loaded.processor, loaded.config, profile.instruction, num_images=1)

    start = time.perf_counter()
    pieces: list[str] = []
    token_probs: list[float] = []
    generation_tokens = 0
    peak_memory_gb = 0.0
    try:
        for chunk in stream_generate(
            loaded.model,
            loaded.processor,
            prompt,
            image=str(image_path),
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            pieces.append(str(getattr(chunk, "text", "")))
            prob = _selected_token_prob(chunk)
            if prob is not None:
                token_probs.append(prob)
            generation_tokens = int(getattr(chunk, "generation_tokens", generation_tokens) or 0)
            peak_memory_gb = float(getattr(chunk, "peak_memory", peak_memory_gb) or peak_memory_gb)
    finally:
        # Release MLX's buffer cache after each run so it doesn't accumulate
        # across many predictions in the long-lived Streamlit process.
        _release_mlx_cache()
    elapsed = time.perf_counter() - start

    caption = "".join(pieces).strip()
    confidence = sum(token_probs) / len(token_probs) if token_probs else None
    disease_id = classify(caption, crop)
    info = _disease_info(disease_id)
    status = _status(disease_id, confidence, confidence_threshold)

    _log.info(
        "prediction crop=%s disease=%s conf=%s status=%s tokens=%d %.2fs caption=%r",
        crop,
        disease_id,
        f"{confidence:.3f}" if confidence is not None else "n/a",
        status,
        generation_tokens,
        elapsed,
        caption[:120],
    )

    return {
        "crop": crop,
        "model": MODEL_ID,
        "adapter": str(profile.adapter_dir),
        "run_name": profile.run_name,
        "instruction": profile.instruction,
        "caption": caption,
        "disease_id": disease_id,
        "disease_name": info["disease_name"],
        "common_name": info["common_name"],
        "symptoms": info["symptoms"],
        "diagnosis": _diagnosis(disease_id, info, crop, status),
        "confidence": confidence,
        "confidence_threshold": confidence_threshold,
        "status": status,
        "inference_seconds": elapsed,
        "inference_time_display": format_seconds(elapsed),
        "generation_tokens": generation_tokens,
        "peak_memory_gb": round(peak_memory_gb, 2) if peak_memory_gb else None,
        "active_memory_gb": _mlx_active_memory_gb(),
    }


def _selected_token_prob(chunk: Any) -> float | None:
    """Probability the model assigned to the token it actually emitted this step."""
    logprobs = getattr(chunk, "logprobs", None)
    token = getattr(chunk, "token", None)
    if logprobs is None or token is None:
        return None
    try:
        import numpy as np

        arr = np.asarray(logprobs).reshape(-1)
        idx = int(token)
        if 0 <= idx < arr.shape[0]:
            return float(math.exp(float(arr[idx])))
    except (TypeError, ValueError, OverflowError):
        return None
    return None


def _status(disease_id: str, confidence: float | None, threshold: float) -> str:
    """Bucket a prediction: named+confident, named+low-confidence, or unknown."""
    if disease_id == UNCLASSIFIED:
        return STATUS_UNKNOWN
    if confidence is not None and confidence < threshold:
        return STATUS_LOW_CONFIDENCE
    return STATUS_CONFIDENT


def _diagnosis(disease_id: str, info: dict[str, Any], crop: str, status: str) -> str:
    """A short, grounded diagnosis line matching the prediction's status."""
    if status == STATUS_UNKNOWN:
        return (
            f"**Unknown.** The model's description of this {crop} leaf did not name "
            "a condition in the knowledge base. This often means the image differs "
            "from the training distribution (PlantVillage-style single-leaf photos). "
            "Review the caption below, or try a clearer, closer single-leaf image."
        )
    name = info["disease_name"]
    common = f" ({info['common_name']})" if info.get("common_name") else ""
    if status == STATUS_LOW_CONFIDENCE:
        return (
            f"**Low confidence.** The model leans toward **{name}**{common}, but its "
            "generation certainty is below the threshold — treat this as tentative. "
            "The image may differ from the training distribution."
        )
    return f"This {crop} leaf is consistent with **{name}**{common}."

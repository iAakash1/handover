"""
Loads an MLX-VLM model and runs a single (image, prompt) -> text
generation, recording wall-clock timestamps, inference time, peak
memory, and (best-effort) token/throughput metrics alongside the raw
text. Every model in config.MODELS goes through this exact same code
path, which is what makes the comparison across models meaningful.
"""

import gc
import time
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, TypedDict

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from mem_utils import get_peak_memory_mb, reset_peak_memory


class GenerationMetrics(TypedDict):
    prompt_tokens: Optional[int]
    generated_tokens: Optional[int]
    prompt_tokens_per_sec: Optional[float]
    generation_tokens_per_sec: Optional[float]
    peak_memory_mb: Optional[float]
    inference_time_sec: float


class GenerationTimestamps(TypedDict):
    start: str
    finish: str
    runtime_sec: float


class GenerationOutcome(TypedDict):
    raw_text: str
    metrics: GenerationMetrics
    timestamps: GenerationTimestamps


def load_model(model_id: str) -> Tuple[Any, Any, Any]:
    """
    Load a model + processor + config once; reuse across all images.

    trust_remote_code is passed defensively: some MLX-VLM model repos
    need it, some don't accept it as a kwarg at all — fall back cleanly
    either way rather than assuming one calling convention.
    """
    try:
        model, processor = load(model_id, trust_remote_code=True)
    except TypeError:
        model, processor = load(model_id)
    model_config = load_config(model_id)
    return model, processor, model_config


def unload_model(model: Any, processor: Any) -> None:
    """Best-effort release of a model's weights before loading the next one."""
    del model, processor
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def _first_attr(obj: Any, *names: str) -> Optional[Any]:
    """Return the value of the first attribute in `names` that exists on
    `obj`, or None if none of them do. Used to read optional metrics off
    mlx_vlm's result object without assuming an exact attribute name,
    since this has varied across mlx_vlm versions."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _extract_metrics(
    result: Any, inference_time_sec: float, peak_memory_mb: Optional[float]
) -> GenerationMetrics:
    """
    Best-effort extraction of token/throughput metrics from mlx_vlm's
    generate() result. Every field is read via getattr with a None
    default: if this installed version of mlx_vlm doesn't expose a given
    metric on its result object, it is stored as null rather than
    estimated, derived, or guessed.
    """
    return {
        "prompt_tokens": _first_attr(result, "prompt_tokens", "num_prompt_tokens"),
        "generated_tokens": _first_attr(
            result, "generation_tokens", "num_generation_tokens", "completion_tokens"
        ),
        "prompt_tokens_per_sec": _first_attr(result, "prompt_tps", "prompt_tokens_per_sec"),
        "generation_tokens_per_sec": _first_attr(
            result, "generation_tps", "generation_tokens_per_sec", "tokens_per_sec"
        ),
        "peak_memory_mb": round(peak_memory_mb, 2) if peak_memory_mb is not None else None,
        "inference_time_sec": round(inference_time_sec, 3),
    }


def generate_response(
    model: Any,
    processor: Any,
    model_config: Any,
    image_path: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> GenerationOutcome:
    """
    Run one generation and return the raw text alongside its metrics and
    timestamps. Peak memory is reset immediately before generation so the
    reading reflects only this single call, not a cumulative high-water
    mark across the whole run.
    """
    formatted_prompt = apply_chat_template(processor, model_config, prompt, num_images=1)

    reset_peak_memory()
    start_dt = datetime.now(timezone.utc)
    start_perf = time.perf_counter()

    result = generate(
        model,
        processor,
        formatted_prompt,
        image=[image_path],
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
    )

    elapsed = time.perf_counter() - start_perf
    finish_dt = datetime.now(timezone.utc)
    peak_mem_mb = get_peak_memory_mb()

    # mlx_vlm's generate() has returned either a plain string or a
    # GenerationResult-like object with a `.text` attribute depending on
    # version — handle both without assuming one or the other.
    text = getattr(result, "text", result)
    if not isinstance(text, str):
        text = str(text)

    return {
        "raw_text": text.strip(),
        "metrics": _extract_metrics(result, elapsed, peak_mem_mb),
        "timestamps": {
            "start": start_dt.isoformat(),
            "finish": finish_dt.isoformat(),
            "runtime_sec": round(elapsed, 3),
        },
    }

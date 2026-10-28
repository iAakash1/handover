"""Diagnose-page inference against the active company adapter (CUDA).

Independent of the MLX path in ``app.inference``: importing this module never
imports mlx, so the app works on an NVIDIA machine. Results use the same record
shape the existing cards render, so no presentation code had to change.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from app.logging_setup import get_logger
from app.utils import DEFAULT_CONFIDENCE_THRESHOLD, format_seconds

STATUS_CONFIDENT = "confident"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_UNKNOWN = "unknown"

_log = get_logger()


@dataclass(frozen=True)
class CompanyHandle:
    """A loaded company model plus how long it took to load."""

    loaded: Any
    cfg: Any
    dataset: Any
    adapter_dir: str | None
    load_seconds: float


def backend_status() -> tuple[bool, str]:
    """Whether this interpreter can run the CUDA company backend, and why not."""
    try:
        import torch
    except Exception as exc:
        return False, f"PyTorch is not importable here ({exc})."
    if not torch.cuda.is_available():
        return False, "PyTorch does not see a CUDA GPU in this interpreter."
    for name in ("transformers", "peft", "bitsandbytes"):
        try:
            __import__(name)
        except Exception as exc:
            return False, f"`{name}` is not importable here ({exc})."
    return True, ""


def active_adapter() -> tuple[Path | None, str]:
    """The adapter the registry points at, plus a short description."""
    try:
        from plantdx.company.predict import resolve_adapter

        return resolve_adapter("tomato")
    except Exception as exc:
        return None, f"registry unreadable: {exc}"


def active_entry() -> Any:
    """The full registry entry for tomato, or ``None``."""
    try:
        from plantdx.company.registry import read_active_adapter

        return read_active_adapter("tomato")
    except Exception:
        return None


@st.cache_resource(show_spinner=False, max_entries=1)
def get_handle(adapter_dir: str | None) -> CompanyHandle:
    """Load (and cache) the base model with ``adapter_dir`` attached."""
    from plantdx.company.classes import load_company_dataset
    from plantdx.company.config import load_company_train_config
    from plantdx.company.modeling import load_for_inference

    cfg = load_company_train_config()
    dataset = load_company_dataset(cfg.data.profile)
    started = time.perf_counter()
    loaded = load_for_inference(cfg, adapter_dir)
    elapsed = time.perf_counter() - started
    _log.info("company model loaded adapter=%s in %.1fs", adapter_dir, elapsed)
    return CompanyHandle(
        loaded=loaded, cfg=cfg, dataset=dataset, adapter_dir=adapter_dir, load_seconds=elapsed
    )


def run_inference(
    handle: CompanyHandle,
    image_path: str | Path,
    *,
    max_tokens: int = 24,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Diagnose one image and return a record the existing result card can render."""
    from plantdx.company.evaluate import UNPARSEABLE
    from plantdx.company.predict import predict_image

    prediction = predict_image(
        handle.loaded,
        image_path,
        cfg=handle.cfg,
        dataset=handle.dataset,
        max_new_tokens=max_tokens,
    )
    status = _status(prediction.class_id, prediction.confidence, confidence_threshold)
    disease_id = prediction.class_id
    if prediction.class_id == UNPARSEABLE:
        disease_id = "unclassified"

    entry = active_entry()
    run_name = entry.run_name if entry else handle.cfg.run_name
    memory = _memory_gb()

    _log.info(
        "company prediction class=%s conf=%s status=%s %.2fs raw=%r",
        prediction.class_id,
        f"{prediction.confidence:.3f}" if prediction.confidence is not None else "n/a",
        status,
        prediction.seconds,
        prediction.generated[:80],
    )

    return {
        "crop": "tomato",
        "model": prediction.base_model,
        "adapter": prediction.adapter or "(base model)",
        "run_name": run_name,
        "instruction": handle.cfg.data.instruction,
        "caption": prediction.generated,
        "disease_id": disease_id,
        "disease_name": prediction.label,
        "common_name": None,
        "symptoms": [],
        "diagnosis": prediction.label,
        "confidence": prediction.confidence,
        "confidence_threshold": confidence_threshold,
        "status": status,
        "inference_seconds": prediction.seconds,
        "inference_time_display": format_seconds(prediction.seconds),
        "generation_tokens": len(prediction.generated.split()),
        "peak_memory_gb": memory,
        "active_memory_gb": memory,
    }


def _memory_gb() -> float | None:
    try:
        from plantdx.company.environment import memory_snapshot

        return memory_snapshot().get("max_reserved_gb")
    except Exception:
        return None


def _status(class_id: str, confidence: float | None, threshold: float) -> str:
    from plantdx.company.evaluate import UNPARSEABLE

    if class_id == UNPARSEABLE:
        return STATUS_UNKNOWN
    if confidence is not None and confidence < threshold:
        return STATUS_LOW_CONFIDENCE
    return STATUS_CONFIDENT


def adapter_summary(adapter_dir: str | Path) -> dict[str, Any]:
    """Rank, target modules, and parameter count of a saved PEFT adapter."""
    path = Path(adapter_dir)
    info: dict[str, Any] = {
        "adapter_dir": str(path),
        "fine_tune_type": None,
        "rank": None,
        "scale": None,
        "num_target_modules": None,
        "trainable_params": None,
        "lora_tensor_count": None,
        "weights_checksum": None,
        "attached": False,
    }
    config_path = path / "adapter_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
        targets = config.get("target_modules") or []
        alpha = config.get("lora_alpha")
        rank = config.get("r")
        info.update(
            fine_tune_type=config.get("peft_type", "LORA"),
            rank=rank,
            scale=(round(alpha / rank, 3) if isinstance(alpha, int) and rank else alpha),
            num_target_modules=len(targets) if isinstance(targets, list) else None,
        )
    weights = path / "adapter_model.safetensors"
    if weights.is_file():
        params, tensors = _count_safetensors(weights)
        info.update(
            trainable_params=params,
            lora_tensor_count=tensors,
            weights_checksum=f"sha256:{_checksum(weights)[:16]}",
            attached=bool(tensors),
        )
    return info


def _count_safetensors(path: Path) -> tuple[int, int]:
    try:
        from safetensors import safe_open

        total, count = 0, 0
        with safe_open(str(path), framework="numpy") as handle:  # type: ignore[no-untyped-call]
            for key in handle.keys():  # noqa: SIM118 - safetensors handle is not a dict
                shape = handle.get_slice(key).get_shape()
                size = 1
                for dim in shape:
                    size *= int(dim)
                total += size
                count += 1
        return total, count
    except Exception:
        return 0, 0


def _checksum(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

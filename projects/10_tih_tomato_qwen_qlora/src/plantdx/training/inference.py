"""Inference with a fine-tuned adapter: single image, a folder, or a batch.

All MLX / mlx-vlm imports are lazy (inside functions) so this module imports
cleanly where MLX is absent (CI, non-Apple hardware). The programmatic API is
``load_model`` + ``caption_image`` / ``caption_folder``; the CLI ``infer`` command
is a thin wrapper over these.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plantdx.core.exceptions import PlantDxError

_DEFAULT_INSTRUCTION = "Describe the visible condition of this tomato leaf."
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


@dataclass(frozen=True)
class CaptionResult:
    """One inference result."""

    image_path: str
    instruction: str
    caption: str


@dataclass
class LoadedModel:
    """A loaded model + processor + config, ready for repeated inference."""

    model: Any
    processor: Any
    config: Any
    model_path: str
    adapter_path: str | None


def load_model(model_path: str, adapter_path: str | None = None) -> LoadedModel:
    """Load the base model (optionally with a trained adapter). Requires mlx-vlm."""
    try:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
    except Exception as exc:  # ABI/env import failures vary (ImportError, AttributeError, ...)
        raise PlantDxError(
            "mlx-vlm is not importable in this environment. Install it on Apple "
            "Silicon (`pip install mlx-vlm`) to run inference."
        ) from exc
    model, processor = load(model_path, adapter_path=adapter_path)
    config = load_config(model_path)
    return LoadedModel(
        model=model,
        processor=processor,
        config=config,
        model_path=model_path,
        adapter_path=adapter_path,
    )


def caption_image(
    loaded: LoadedModel,
    image_path: str | Path,
    *,
    instruction: str = _DEFAULT_INSTRUCTION,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> CaptionResult:
    """Caption one image. Deterministic by default (temperature 0)."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(loaded.processor, loaded.config, instruction, num_images=1)
    result = generate(
        loaded.model,
        loaded.processor,
        prompt,
        image=str(image_path),
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
    )
    text = getattr(result, "text", str(result)).strip()
    return CaptionResult(image_path=str(image_path), instruction=instruction, caption=text)


def caption_folder(
    loaded: LoadedModel,
    folder: str | Path,
    *,
    instruction: str = _DEFAULT_INSTRUCTION,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> list[CaptionResult]:
    """Caption every image in ``folder`` (non-recursive), in sorted order."""
    paths = discover_image_paths(folder)
    if not paths:
        raise PlantDxError(f"no images found in {folder}")
    return [
        caption_image(
            loaded, p, instruction=instruction, max_tokens=max_tokens, temperature=temperature
        )
        for p in paths
    ]


def caption_batch(
    loaded: LoadedModel,
    image_paths: list[str | Path],
    *,
    instruction: str = _DEFAULT_INSTRUCTION,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> list[CaptionResult]:
    """Caption an explicit list of images."""
    return [
        caption_image(
            loaded, p, instruction=instruction, max_tokens=max_tokens, temperature=temperature
        )
        for p in image_paths
    ]


def discover_image_paths(folder: str | Path) -> list[Path]:
    """Return sorted image paths directly under ``folder``."""
    root = Path(folder)
    if not root.is_dir():
        raise PlantDxError(f"not a directory: {folder}")
    return sorted(p for p in root.iterdir() if p.suffix in _IMAGE_SUFFIXES and p.is_file())

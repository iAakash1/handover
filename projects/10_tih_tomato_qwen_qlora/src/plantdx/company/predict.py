"""Single-image prediction with a trained company adapter.

Resolves which adapter to serve from the active-adapter registry, loads it once,
and returns a parsed class plus the raw generation and a generation confidence.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plantdx.company.classes import CompanyDataset, load_company_dataset
from plantdx.company.config import CompanyTrainConfig, load_company_train_config
from plantdx.company.evaluate import UNPARSEABLE, parse_prediction
from plantdx.company.modeling import (
    LoadedModel,
    ModelingError,
    build_prompt,
    load_for_inference,
    load_image,
)
from plantdx.company.registry import read_active_adapter


@dataclass(frozen=True)
class Prediction:
    """One prediction and how it was produced."""

    class_id: str
    label: str
    generated: str
    confidence: float | None
    seconds: float
    adapter: str | None
    base_model: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return asdict(self)


def resolve_adapter(
    crop: str = "tomato", *, base_dir: str | Path | None = None
) -> tuple[Path | None, str]:
    """Return the active adapter directory for ``crop`` and a description.

    Falls back to ``(None, reason)`` when nothing is activated or the recorded
    directory has gone missing, so the caller can serve the base model or show a
    clear message instead of failing.
    """
    entry = read_active_adapter(crop, base_dir)
    if entry is None:
        return None, "no active adapter registered"
    path = entry.resolved_path(base_dir)
    if not path.is_dir():
        return None, f"registered adapter is missing: {path}"
    return path, f"{entry.run_name} ({entry.dataset})"


def load_predictor(
    cfg: CompanyTrainConfig | None = None,
    *,
    adapter_dir: str | Path | None = None,
    crop: str = "tomato",
    local_files_only: bool = False,
) -> LoadedModel:
    """Load the model with the requested (or active) adapter attached."""
    cfg = cfg or load_company_train_config()
    if adapter_dir is None:
        resolved, _ = resolve_adapter(crop, base_dir=cfg.base_dir)
        adapter_dir = resolved
    return load_for_inference(cfg, adapter_dir, local_files_only=local_files_only)


def predict_image(
    loaded: LoadedModel,
    image_path: str | Path,
    *,
    cfg: CompanyTrainConfig | None = None,
    dataset: CompanyDataset | None = None,
    max_new_tokens: int | None = None,
) -> Prediction:
    """Diagnose one image and parse the answer back to a class."""
    import torch

    cfg = cfg or load_company_train_config()
    dataset = dataset or load_company_dataset(cfg.data.profile)

    image = load_image(image_path, cfg.model.image_max_side)
    prompt = build_prompt(loaded.processor, cfg.data.instruction)
    inputs = loaded.processor(text=[prompt], images=[image], return_tensors="pt")
    device = next(loaded.model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    started = time.perf_counter()
    with torch.no_grad():
        output = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or cfg.evaluation.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=loaded.processor.tokenizer.pad_token_id,
        )
    elapsed = time.perf_counter() - started

    prompt_len = int(inputs["input_ids"].shape[1])
    tokens = output.sequences[0][prompt_len:]
    text = str(loaded.processor.tokenizer.decode(tokens, skip_special_tokens=True)).strip()
    confidence = _confidence(output, tokens)

    class_id = parse_prediction(text, dataset)
    entry = dataset.by_class_id(class_id)
    return Prediction(
        class_id=class_id,
        label=entry.label if entry else "Unrecognized",
        generated=text,
        confidence=confidence,
        seconds=round(elapsed, 3),
        adapter=loaded.adapter_path,
        base_model=loaded.model_id,
    )


def _confidence(output: Any, tokens: Any) -> float | None:
    """Mean probability of the tokens the model actually emitted."""
    import torch

    scores = getattr(output, "scores", None)
    if not scores:
        return None
    try:
        probabilities: list[float] = []
        for step, logits in enumerate(scores):
            if step >= tokens.shape[0]:
                break
            distribution = torch.softmax(logits[0].float(), dim=-1)
            probabilities.append(float(distribution[int(tokens[step])]))
        if not probabilities:
            return None
        return round(sum(probabilities) / len(probabilities), 4)
    except (IndexError, RuntimeError, ValueError):
        return None


def predict_paths(
    paths: list[str | Path],
    *,
    cfg: CompanyTrainConfig | None = None,
    adapter_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[Prediction]:
    """Diagnose several images with one model load.

    Raises:
        ModelingError: No images were given.
    """
    if not paths:
        raise ModelingError("no images to predict")
    cfg = cfg or load_company_train_config()
    dataset = load_company_dataset(cfg.data.profile)
    loaded = load_predictor(cfg, adapter_dir=adapter_dir, local_files_only=local_files_only)
    return [predict_image(loaded, p, cfg=cfg, dataset=dataset) for p in paths]


def is_unparsed(prediction: Prediction) -> bool:
    """Whether the generation did not name a known class."""
    return prediction.class_id == UNPARSEABLE

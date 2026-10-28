"""Model, processor, and batching for the company Qwen2.5-VL QLoRA run.

One place builds the 4-bit base model, attaches the LoRA adapter, and turns a
``{"image", "question", "answer"}`` row into tensors. Training, the smoke test,
evaluation, and single-image prediction all go through these functions, so the
Train button exercises exactly the path the smoke test proved.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plantdx.company.config import CompanyTrainConfig
from plantdx.core.exceptions import PlantDxError

PATCH = 28  # Qwen2.5-VL vision patch size; every side is rounded to a multiple.
IGNORE_INDEX = -100
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


class ModelingError(PlantDxError):
    """The model, processor, or a batch could not be built."""


@dataclass
class LoadedModel:
    """A ready-to-use model plus its processor and a short description."""

    model: Any
    processor: Any
    model_id: str
    adapter_path: str | None
    trainable_params: int
    total_params: int


def torch_dtype(name: str) -> Any:
    """Map a config dtype name onto a torch dtype."""
    import torch

    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ModelingError(f"unsupported dtype {name!r}")
    return mapping[name]


def load_processor(cfg: CompanyTrainConfig, *, local_files_only: bool = False) -> Any:
    """Load the Qwen processor for the configured base model."""
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            cfg.model.model_id, local_files_only=local_files_only
        )
    except Exception as exc:
        raise ModelingError(f"could not load processor for {cfg.model.model_id}: {exc}") from exc


def load_base_model(
    cfg: CompanyTrainConfig,
    *,
    local_files_only: bool = False,
    for_training: bool = True,
) -> Any:
    """Load Qwen2.5-VL, quantized to 4-bit when the config asks for it."""
    import torch
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    compute_dtype = torch_dtype(cfg.model.compute_dtype)
    kwargs: dict[str, Any] = {
        "dtype": compute_dtype,
        "local_files_only": local_files_only,
    }
    if cfg.model.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.model.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if not torch.cuda.is_available():
        raise ModelingError("Company QLoRA requires CUDA")
    kwargs["device_map"] = {"": 0}

    try:
        model = AutoModelForImageTextToText.from_pretrained(cfg.model.model_id, **kwargs)
    except Exception as exc:
        raise ModelingError(f"could not load {cfg.model.model_id}: {exc}") from exc

    model.config.use_cache = not for_training
    return model


def language_model_targets(model: Any, wanted: Sequence[str]) -> list[str]:
    """Full module names of the language-model projections to adapt.

    Selecting by full name (rather than by bare suffix) is what keeps the vision
    tower frozen: Qwen's visual blocks carry ``gate_proj``/``up_proj``/
    ``down_proj`` names too, and a suffix match would adapt them as well.
    """
    import torch.nn as nn

    suffixes = set(wanted)
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) and module.__class__.__name__ != "Linear4bit":
            continue
        if "visual" in name or "vision" in name:
            continue
        if name.rsplit(".", 1)[-1] in suffixes:
            targets.append(name)
    return targets


def vision_targets(model: Any, wanted: Sequence[str]) -> list[str]:
    """Full module names of vision-tower projections matching ``wanted``."""
    import torch.nn as nn

    suffixes = set(wanted)
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) and module.__class__.__name__ != "Linear4bit":
            continue
        if "visual" not in name and "vision" not in name:
            continue
        if name.rsplit(".", 1)[-1] in suffixes:
            targets.append(name)
    return targets


def attach_lora(model: Any, cfg: CompanyTrainConfig) -> Any:
    """Prepare the quantized model for training and attach the LoRA adapter."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    targets = language_model_targets(model, cfg.lora.target_modules)
    if cfg.lora.train_vision:
        raise ModelingError("Company training keeps the vision tower frozen")
    if not targets:
        raise ModelingError(
            f"no modules matched lora.target_modules={list(cfg.lora.target_modules)}"
        )

    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
            model, use_gradient_checkpointing=cfg.optim.grad_checkpoint
        )
    if cfg.optim.grad_checkpoint:
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    return get_peft_model(model, lora_config)


def count_params(model: Any) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def resize_image(image: Any, max_side: int) -> Any:
    """Aspect-preserving resize into a ``max_side`` box, rounded to the patch grid.

    Bounds the visual-token count without the distortion a square resize causes
    on the portrait photographs in this dataset.
    """
    from PIL import Image

    width, height = image.size
    scale = max_side / max(width, height)
    new_w = max(PATCH, round(width * scale / PATCH) * PATCH)
    new_h = max(PATCH, round(height * scale / PATCH) * PATCH)
    if (new_w, new_h) == (width, height):
        return image
    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


def load_image(path: str | Path, max_side: int) -> Any:
    """Open an image as RGB and resize it for the vision tower."""
    from PIL import Image

    try:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
    except Exception as exc:
        raise ModelingError(f"could not read image {path}: {exc}") from exc
    return resize_image(image, max_side)


def build_prompt(processor: Any, question: str) -> str:
    """The chat prompt up to and including the assistant turn marker."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(text)


class JsonlRows:
    """A torch-compatible dataset over one prepared JSONL split."""

    def __init__(self, path: str | Path, *, limit: int | None = None) -> None:
        """Read every row from ``path``, optionally keeping only the first ``limit``."""
        source = Path(path)
        if not source.is_file():
            raise ModelingError(
                f"dataset split not found: {source} (run `plantdx company prepare`)"
            )
        rows: list[dict[str, Any]] = []
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
        if not rows:
            raise ModelingError(f"dataset split is empty: {source}")
        self.rows = rows
        self.path = source

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Row at ``index``."""
        return self.rows[index]


class CompanyCollator:
    """Turn rows into a batch with the prompt masked out of the loss.

    The prompt is tokenized once on its own and once with the answer appended;
    the shared token prefix is set to ``-100`` so the loss covers only the
    answer. The common-prefix length is measured rather than assumed, so a
    tokenizer that merges across the boundary cannot silently unmask the prompt.
    """

    def __init__(self, processor: Any, *, max_side: int, max_seq_length: int) -> None:
        """Bind the processor and the image/sequence bounds."""
        self.processor = processor
        self.max_side = max_side
        self.max_seq_length = max_seq_length

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Collate ``batch`` into model inputs with masked labels."""
        import torch

        features: list[dict[str, Any]] = []
        for row in batch:
            features.append(self._encode(row))

        max_len = max(f["input_ids"].shape[0] for f in features)
        pad_id = int(self.processor.tokenizer.pad_token_id)

        input_ids = torch.full((len(features), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        labels = torch.full((len(features), max_len), IGNORE_INDEX, dtype=torch.long)
        for i, feature in enumerate(features):
            length = feature["input_ids"].shape[0]
            input_ids[i, :length] = feature["input_ids"]
            attention_mask[i, :length] = feature["attention_mask"]
            labels[i, :length] = feature["labels"]

        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat([f["pixel_values"] for f in features], dim=0),
            "image_grid_thw": torch.cat([f["image_grid_thw"] for f in features], dim=0),
        }
        return out

    def _encode(self, row: dict[str, Any]) -> dict[str, Any]:
        image = load_image(row["image"], self.max_side)
        prompt = build_prompt(self.processor, str(row["question"]))
        answer = str(row["answer"])
        full = f"{prompt}{answer}<|im_end|>\n"

        encoded = self.processor(text=[full], images=[image], return_tensors="pt")
        prompt_only = self.processor(text=[prompt], images=[image], return_tensors="pt")

        input_ids = encoded["input_ids"][0]
        prompt_ids = prompt_only["input_ids"][0]
        prefix = _common_prefix(input_ids.tolist(), prompt_ids.tolist())

        labels = input_ids.clone()
        labels[:prefix] = IGNORE_INDEX
        attention_mask = encoded["attention_mask"][0]

        if input_ids.shape[0] > self.max_seq_length:
            raise ModelingError(
                f"sequence of {input_ids.shape[0]} tokens exceeds max_seq_length="
                f"{self.max_seq_length} for {row['image']}; lower model.image_max_side "
                "or raise model.max_seq_length"
            )
        if int((labels != IGNORE_INDEX).sum()) == 0:
            raise ModelingError(f"every label was masked for {row['image']}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
        }


def _common_prefix(a: list[int], b: list[int]) -> int:
    count = 0
    for left, right in zip(a, b, strict=False):
        if left != right:
            break
        count += 1
    return count


def adapter_is_valid(adapter_dir: str | Path) -> tuple[bool, str]:
    """Whether ``adapter_dir`` holds a loadable PEFT adapter, and why not if it does not."""
    path = Path(adapter_dir)
    if not path.is_dir():
        return False, f"adapter directory not found: {path}"
    for name in ADAPTER_FILES:
        if not (path / name).is_file():
            return False, f"missing {name} in {path}"
    try:
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"unreadable adapter_config.json: {exc}"
    if not config.get("target_modules"):
        return False, "adapter_config.json has no target_modules"
    return True, ""


def load_for_inference(
    cfg: CompanyTrainConfig,
    adapter_dir: str | Path | None,
    *,
    local_files_only: bool = False,
) -> LoadedModel:
    """Load the base model, optionally attach a trained adapter, and set eval mode."""
    processor = load_processor(cfg, local_files_only=local_files_only)
    model = load_base_model(cfg, local_files_only=local_files_only, for_training=False)

    adapter_path: str | None = None
    if adapter_dir is not None:
        ok, reason = adapter_is_valid(adapter_dir)
        if not ok:
            raise ModelingError(reason)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        adapter_path = str(adapter_dir)

    model.eval()
    trainable, total = count_params(model)
    return LoadedModel(
        model=model,
        processor=processor,
        model_id=cfg.model.model_id,
        adapter_path=adapter_path,
        trainable_params=trainable,
        total_params=total,
    )


def generate_label(
    loaded: LoadedModel,
    image_path: str | Path,
    question: str,
    *,
    max_side: int,
    max_new_tokens: int,
) -> str:
    """Greedy-decode one answer for one image."""
    import torch

    image = load_image(image_path, max_side)
    prompt = build_prompt(loaded.processor, question)
    inputs = loaded.processor(text=[prompt], images=[image], return_tensors="pt")
    device = next(loaded.model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        generated = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=loaded.processor.tokenizer.pad_token_id,
        )
    prompt_len = int(inputs["input_ids"].shape[1])
    text = loaded.processor.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    return str(text).strip()

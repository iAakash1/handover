"""
The VLM adapter contract.

WHY THIS EXISTS
    Model-specific knowledge is unavoidable — every vision-language model has its
    own chat markers, its own image-token bookkeeping, its own class name, its
    own idea of what "resize the image" means. The mistake is letting that
    knowledge spread. In the previous design, Qwen-specific details lived in the
    dataset module, in the trainer, and in the inference script, so supporting a
    second model meant editing three files and hoping you found every place.

    Here, everything a model family needs to know lives behind this one
    interface. The trainer, the dataset, and the inference engine are written
    against the interface and contain no model-specific code at all.

TO ADD A NEW VLM
    1. Write a class implementing VLMAdapter (copy qwen2_5_vl.py as a template).
    2. Add it to ADAPTERS in plantvlm/models/__init__.py.
    3. Set model.adapter in the YAML config.
    Nothing else in the project changes.

THE CONTRACT IN WORDS
    build_train_sample   (image, instruction, caption) -> tensors, with the loss
                         masked to the model's own answer turn
    build_generate_inputs(image, instruction)          -> tensors ready to sample
    decode_new_tokens    generated ids                 -> the text the model wrote
    lora_targets         the loaded model              -> which Linear layers to adapt
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import torch
from PIL import Image


@runtime_checkable
class VLMAdapter(Protocol):
    """One implementation per model family."""

    name: str                # registry key, e.g. "qwen2_5_vl"
    assistant_marker: str    # chat token that begins the model's own turn
    dtype: torch.dtype       # resolved compute dtype (used by autocast)
    device: str              # "cuda" or "cpu"

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_processor(self, source: Optional[str] = None) -> Any:
        """Processor/tokenizer, configured with the project's vision resolution."""
        ...

    def load_model(self, source: Optional[str] = None,
                   for_training: bool = False) -> Any:
        """The base model on device, in the configured dtype."""
        ...

    # ── Training ─────────────────────────────────────────────────────────────

    def build_train_sample(self, image: Image.Image, instruction: str,
                           caption: str, processor: Any,
                           max_length: int) -> Dict[str, torch.Tensor]:
        """
        One supervised example.

        MUST return labels where every token that is not part of the model's own
        answer is set to -100. The model is given the image and the question for
        free; it is scored only on the caption.
        """
        ...

    def collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack samples into a batch."""
        ...

    def lora_targets(self, model: Any, suffixes: List[str],
                     exclude_substring: str) -> List[str]:
        """
        Fully-qualified names of the Linear layers LoRA should adapt.

        Implementations must exclude the vision tower unless explicitly asked
        otherwise: adapting an image encoder needs far more data than a
        caption-scale dataset provides.
        """
        ...

    # ── Inference ────────────────────────────────────────────────────────────

    def build_generate_inputs(self, image: Image.Image, instruction: str,
                              processor: Any, device: str) -> Dict[str, torch.Tensor]:
        """Inputs for a single-image generation call."""
        ...

    def decode_new_tokens(self, processor: Any, sequences: torch.Tensor,
                          prompt_length: int) -> str:
        """Text of the newly generated tokens only, prompt excluded."""
        ...

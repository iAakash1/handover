"""
Qwen2.5-VL adapter — the only file in the project that knows what Qwen is.

WHAT IS QWEN-SPECIFIC AND THEREFORE LIVES HERE
    * the model class name (Qwen2_5_VLForConditionalGeneration)
    * min_pixels / max_pixels, and the fact that they are expressed in 28x28
      patches which are then merged 2x2 into visual tokens
    * the chat template and the "<|im_start|>assistant" marker that tells us
      where the answer begins, so the loss can be masked to it
    * image_grid_thw, which the model requires alongside pixel_values
    * process_vision_info from qwen_vl_utils

VISION RESOLUTION — WHY IT IS SET EXPLICITLY
    The source images are 256x256. At the processor's defaults that becomes ~81
    visual tokens, which is a coarse grid for captions that say "near the upper
    left section" or "small pale specks". min_pixels_factor=256 forces ~448x448
    and ~256 visual tokens, giving the language model a finer grid to ground
    spatial language on. Training and inference read the same config value, so
    they can never disagree.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor

from ..config import Config

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover
    raise ImportError("Qwen2.5-VL needs qwen-vl-utils: pip install qwen-vl-utils"
                      ) from exc


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
           "float32": torch.float32}


class Qwen25VLAdapter:
    """Implements plantvlm.models.base.VLMAdapter for the Qwen2.5-VL family."""

    name = "qwen2_5_vl"
    assistant_marker = "<|im_start|>assistant"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.dtype = _DTYPES[cfg.model.dtype]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._marker_ids: Optional[List[int]] = None

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_processor(self, source: Optional[str] = None) -> Any:
        return AutoProcessor.from_pretrained(
            source or self.cfg.model.name,
            min_pixels=self.cfg.min_pixels,
            max_pixels=self.cfg.max_pixels,
            padding_side="left",
        )

    def load_model(self, source: Optional[str] = None,
                   for_training: bool = False) -> Any:
        from transformers import Qwen2_5_VLForConditionalGeneration

        name = source or self.cfg.model.name
        try:                                    # transformers >= 5
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                name, dtype=self.dtype, device_map="auto")
        except TypeError:                       # transformers 4.x
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                name, torch_dtype=self.dtype, device_map="auto")

        if for_training:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        else:
            model.config.use_cache = True
            model.eval()
        return model

    # ── Conversation ─────────────────────────────────────────────────────────

    @staticmethod
    def _messages(image: Image.Image, instruction: str,
                  caption: Optional[str] = None) -> List[Dict]:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image},
                        {"type": "text", "text": instruction}],
        }]
        if caption is not None:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            })
        return messages

    def _marker(self, processor: Any) -> List[int]:
        if self._marker_ids is None:
            self._marker_ids = processor.tokenizer.encode(
                self.assistant_marker, add_special_tokens=False)
        return self._marker_ids

    def _answer_start(self, ids: torch.Tensor, processor: Any) -> int:
        """
        Index of the first token after the assistant marker.

        Found by scanning input_ids rather than measuring prompt length, because
        the processor injects a variable number of image tokens that the
        tokenizer alone cannot account for.
        """
        marker = self._marker(processor)
        n = len(marker)
        flat = ids.tolist()
        for i in range(len(flat) - n):
            if flat[i:i + n] == marker:
                return i + n
        return -1

    # ── Training ─────────────────────────────────────────────────────────────

    def build_train_sample(self, image: Image.Image, instruction: str,
                           caption: str, processor: Any,
                           max_length: int) -> Dict[str, torch.Tensor]:
        messages = self._messages(image, instruction, caption)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        images, _ = process_vision_info(messages)

        enc = processor(text=[text], images=images, padding="max_length",
                        max_length=max_length, truncation=True,
                        return_tensors="pt")

        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()

        start = self._answer_start(input_ids, processor)
        if start < 0:
            raise RuntimeError(
                "assistant marker not found — the chat template changed and "
                "label masking would silently supervise the wrong tokens")
        labels[:start] = -100
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": enc["pixel_values"].squeeze(0),
            "image_grid_thw": enc["image_grid_thw"].squeeze(0),
            "labels": labels,
        }

    def collate(self, batch: List[Dict[str, torch.Tensor]]
                ) -> Dict[str, torch.Tensor]:
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

    def lora_targets(self, model: Any, suffixes: List[str],
                     exclude_substring: str) -> List[str]:
        targets = [
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name.split(".")[-1] in suffixes
            and exclude_substring not in name
        ]
        if not targets:
            raise RuntimeError(
                f"no LoRA targets matched {suffixes} — the model layout changed")
        return targets

    # ── Inference ────────────────────────────────────────────────────────────

    def build_generate_inputs(self, image: Image.Image, instruction: str,
                              processor: Any, device: str
                              ) -> Dict[str, torch.Tensor]:
        messages = self._messages(image, instruction, caption=None)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(messages)
        # Single sequence -> no padding (padding=True makes transformers warn
        # about an ignored max_length).
        inputs = processor(text=[text], images=images, padding=False,
                           return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    def decode_new_tokens(self, processor: Any, sequences: torch.Tensor,
                          prompt_length: int) -> str:
        return processor.decode(sequences[0][prompt_length:],
                                skip_special_tokens=True).strip()

    # ── Introspection (used by the verifier and the smoke test) ──────────────

    def visual_tokens(self, sample: Dict[str, torch.Tensor]) -> int:
        """Visual tokens implied by image_grid_thw (2x2 patch merge)."""
        return int(sample["image_grid_thw"].prod() // 4)

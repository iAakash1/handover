"""
Stage 6 — Conversation formatting at training time.

WHY THIS IS NOT A "STAGE" THAT PRODUCES A FILE
    Conversation formatting is often drawn as a box between "JSONL" and
    "training", as if it produced another artifact. It does not. A conversation
    is built per sample, inside the dataloader, every epoch, and it depends on
    the model. Drawing it as a persisted stage is a small lie that leads to a
    real bug: people format once with one processor and train with another.

    Here it is exactly what it is — the runtime bridge between the JSONL contract
    and whichever model is being trained, delegated entirely to the adapter.

INPUT   JSONL rows (image path + validated caption + instruction)
OUTPUT  batches of tensors with the loss masked to the answer turn

WHY THERE IS NO AUGMENTATION
    The validated captions make absolute spatial claims: "concentrated along the
    right margin", "near the upper left section", "the tip appears darker". A
    horizontal flip makes the caption false. A random crop can delete the feature
    being described. Colour jitter breaks "bright yellow-green" versus "deeper
    green". With 150 hand-written captions, corrupting the supervision is a far
    worse trade than the variety augmentation would buy — so the flag exists in
    config, defaults to False, and raises if switched on without a spatial-safe
    transform being supplied.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..models.base import VLMAdapter


class CaptionDataset(Dataset):
    """
    One JSONL row -> one supervised example.

    The five captions of an image are five independent rows, shuffled with
    everything else. The model therefore meets the same pixels several times per
    epoch paired with different valid answers, and no single string can minimise
    the loss. That is the core regularisation of this project and the reason the
    "flashcard" framing is technically correct, not just a teaching metaphor.
    """

    def __init__(self, rows: List[Dict], adapter: VLMAdapter, processor,
                 cfg: Config) -> None:
        if cfg.train.augment:
            raise ValueError(
                "train.augment=True but no spatial-safe transform is defined. "
                "The validated captions reference absolute positions; flips, "
                "rotations and crops would falsify them. See dataset docstring.")
        self.rows = rows
        self.adapter = adapter
        self.processor = processor
        self.max_length = cfg.model.max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        image = Image.open(row["image"]).convert("RGB")
        return self.adapter.build_train_sample(
            image=image,
            instruction=row["instruction"],
            caption=row["caption"],
            processor=self.processor,
            max_length=self.max_length,
        )


def build_loader(rows: List[Dict], adapter: VLMAdapter, processor,
                 cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        CaptionDataset(rows, adapter, processor, cfg),
        batch_size=cfg.train.micro_batch,
        shuffle=shuffle,
        collate_fn=adapter.collate,
        num_workers=cfg.train.num_workers,   # 0 on Windows
        pin_memory=True,
    )


def describe_sample(sample: Dict[str, torch.Tensor], processor) -> Dict:
    """Introspection used by the verifier and the smoke test."""
    labels = sample["labels"]
    supervised = labels[labels != -100]
    return {
        "tokens_used": int(sample["attention_mask"].sum()),
        "supervised_tokens": int(supervised.numel()),
        "image_grid_thw": tuple(sample["image_grid_thw"].tolist()),
        "supervised_text": processor.tokenizer.decode(
            supervised, skip_special_tokens=True).strip(),
    }

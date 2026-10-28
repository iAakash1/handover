"""
Stage 7 — LoRA fine-tuning.

WHY LoRA AND NOT FULL FINE-TUNING
    Two independent reasons, and either alone would be sufficient. A 7B model's
    weights plus optimizer state do not fit in 24 GB. And 18 training images
    cannot responsibly move 7 billion parameters — the capacity has to match the
    evidence. LoRA freezes the base model and trains a small set of new weights,
    so the checkpoint is megabytes, the run is minutes, and the original model is
    provably unchanged.

WHY THE VISION TOWER IS NEVER ADAPTED
    An image encoder learns what edges, textures and colours mean. That is not
    learnable from eighteen photographs. The adapter builds the target list by
    walking the model and excluding anything in the vision tower, and this
    trainer asserts the result before the first forward pass — if a visual
    parameter is trainable, it refuses to start rather than quietly wasting a run.

INPUT   train/val JSONL, a Config, a Run directory
OUTPUT  checkpoints, per-step and per-epoch logs, the best adapter by val loss

CONNECTS TO
    Inference loads run.best and generates on held-out images.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from ..config import Config
from ..io import append_jsonl, write_json
from ..models.base import VLMAdapter
from .dataset import build_loader
from .run import Run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainResult:
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    minutes: float
    early_stopped: bool


class Trainer:
    """
    One fine-tuning run: optimizer, schedule, accumulation, validation,
    checkpointing, early stopping.

    Deliberately contains no model-specific code — everything that knows what a
    Qwen is lives behind the adapter, so this class works unchanged for the next
    VLM.
    """

    def __init__(self, cfg: Config, adapter: VLMAdapter, run: Run,
                 train_rows: List[Dict], val_rows: List[Dict],
                 resume_from: Optional[str] = None) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.run = run
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resume_from = resume_from

        set_seed(cfg.seed)

        self.processor = adapter.load_processor(resume_from)
        self.model = adapter.load_model(resume_from, for_training=True)
        self.model = self._attach_lora()

        self.train_loader = build_loader(train_rows, adapter, self.processor,
                                         cfg, shuffle=True)
        self.val_loader = build_loader(val_rows, adapter, self.processor,
                                       cfg, shuffle=False)

        self.steps_per_epoch = max(len(self.train_loader) // cfg.train.grad_accum, 1)
        self.total_steps = self.steps_per_epoch * cfg.train.epochs
        warmup = max(int(self.total_steps * cfg.train.warmup_ratio), 1)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup,
            num_training_steps=self.total_steps)
        self._trainable = trainable

        self.global_step = 0
        self.best_val = float("inf")
        self.best_epoch = 0

    # ── Setup ────────────────────────────────────────────────────────────────

    def _attach_lora(self):
        from peft import LoraConfig, PeftModel, get_peft_model

        lo = self.cfg.lora
        if self.resume_from:
            model = PeftModel.from_pretrained(
                self.model, self.resume_from, is_trainable=True)
        else:
            targets = self.adapter.lora_targets(
                self.model, lo.target_suffixes, lo.exclude_substring)
            model = get_peft_model(self.model, LoraConfig(
                r=lo.r, lora_alpha=lo.alpha, lora_dropout=lo.dropout,
                target_modules=targets, bias="none", task_type="CAUSAL_LM"))
            self.n_targets = len(targets)

        leaked = [n for n, p in model.named_parameters()
                  if p.requires_grad and self.cfg.lora.exclude_substring in n]
        if leaked:
            raise RuntimeError(
                f"{len(leaked)} vision-tower parameters are trainable "
                f"(first: {leaked[0]}). Refusing to adapt an image encoder on "
                f"{len(self.train_loader.dataset)} rows.")
        return model

    def preflight(self) -> Dict:
        """
        Cheap assertions that catch the failures which otherwise look like a
        normal loss curve.
        """
        sample = self.train_loader.dataset[0]
        supervised = int((sample["labels"] != -100).sum())
        used = int(sample["attention_mask"].sum())
        if supervised < 10:
            raise RuntimeError(
                f"only {supervised} supervised tokens — label masking is broken")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "supervised_tokens": supervised,
            "tokens_used": used,
            "max_length": self.cfg.model.max_length,
            "truncating": used >= self.cfg.model.max_length,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
        }

    # ── Loops ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total = 0.0
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.autocast(self.device, dtype=self.adapter.dtype):
                total += self.model(**batch).loss.item()
        return total / max(len(self.val_loader), 1)

    def _train_epoch(self, epoch: int, verbose: bool = True) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        running, n = 0.0, 0

        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.autocast(self.device, dtype=self.adapter.dtype):
                loss = self.model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}")
            (loss / self.cfg.train.grad_accum).backward()

            running += loss.item()
            n += 1

            if (step + 1) % self.cfg.train.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._trainable, self.cfg.train.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                vram = (torch.cuda.memory_allocated() / 1e9
                        if self.device == "cuda" else 0.0)
                append_jsonl(self.run.logs / "train.jsonl", {
                    "epoch": epoch, "step": self.global_step,
                    "loss": round(running / n, 5),
                    "lr": self.scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm), "vram_gb": round(vram, 2)})
                if verbose:
                    print(f"  epoch {epoch}/{self.cfg.train.epochs}  "
                          f"step {self.global_step}/{self.total_steps}  "
                          f"loss {running / n:.4f}  "
                          f"lr {self.scheduler.get_last_lr()[0]:.2e}  "
                          f"vram {vram:.1f} GB")

        return running / max(n, 1)

    def save(self, directory, meta: Dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(directory))
        self.processor.save_pretrained(str(directory))
        write_json(meta, directory / "checkpoint_meta.json")

    def fit(self, verbose: bool = True) -> TrainResult:
        started = time.time()
        patience = 0
        early = False

        for epoch in range(1, self.cfg.train.epochs + 1):
            train_loss = self._train_epoch(epoch, verbose)
            val_loss = self.validate()
            if verbose:
                print(f"\n  epoch {epoch}: train {train_loss:.4f} | "
                      f"val {val_loss:.4f}")

            append_jsonl(self.run.logs / "epochs.jsonl", {
                "epoch": epoch, "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5)})

            meta = {"run": self.run.name, "epoch": epoch,
                    "global_step": self.global_step,
                    "train_loss": train_loss, "val_loss": val_loss,
                    "config": self.cfg.to_dict()}

            if self.cfg.train.save_every_epoch:
                self.save(self.run.checkpoint(epoch), meta)

            if val_loss < self.best_val:
                self.best_val, self.best_epoch, patience = val_loss, epoch, 0
                self.save(self.run.best, {**meta, "is_best": True})
                if verbose:
                    print(f"  new best -> {self.run.best}\n")
            else:
                patience += 1
                if verbose:
                    print(f"  no improvement ({patience}/{self.cfg.train.patience})\n")
                if patience >= self.cfg.train.patience:
                    early = True
                    if verbose:
                        print("  early stopping\n")
                    break

        return TrainResult(
            best_val_loss=self.best_val, best_epoch=self.best_epoch,
            epochs_run=epoch, minutes=(time.time() - started) / 60,
            early_stopped=early)

# 02_finetune_qwen25vl.py
# Fine-tunes Qwen2.5-VL-7B-Instruct on PlantVillage using:
#   - 4-bit QLoRA (BitsAndBytes NF4) for minimal VRAM footprint
#   - Flash Attention 2 for 40% faster attention
#   - LoRA rank 64 on all linear layers
#   - Effective batch size 32 (bs=4, grad_accum=8)
#   - Cosine LR schedule with warmup
#
# Expected VRAM: ~20-22 GB peak on RTX 4500 Ada 24GB
# Expected runtime: ~60-90 min for 3 epochs on 20k images

import json
import torch
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
import torchvision.transforms as T

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-VL-7B-Instruct"
CAPTIONS     = "captions.json"
OUTPUT_DIR   = "outputs/qwen25vl_plantvillage"
EPOCHS       = 3
BATCH_SIZE   = 4           # fits in ~22GB with 4-bit quant
GRAD_ACCUM   = 8           # effective batch = 32
LR           = 2e-4
MAX_PIXELS   = 512 * 28 * 28   # Qwen2.5-VL uses pixel budget, not fixed size
MIN_PIXELS   = 256 * 28 * 28
MAX_LENGTH   = 256         # tokens for caption
LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05
SAVE_STEPS   = 300
LOG_STEPS    = 20
PATIENCE     = 3
# ─────────────────────────────────────────────────────────────────────────

# ── Augmentation pipeline ─────────────────────────────────────────────────
AUGMENT = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.08),
    T.RandomResizedCrop(size=(448, 448), scale=(0.75, 1.0)),
    T.RandomGrayscale(p=0.05),
])


class PlantVillageDataset(Dataset):
    """
    Formats each sample as a Qwen2.5-VL chat message:
      [
        {"role": "user",      "content": [<image>, <text>]},
        {"role": "assistant", "content": <caption>},
      ]
    The processor converts this to input_ids + pixel_values.
    Labels mask the user portion so loss is computed only on the caption.
    """
    def __init__(self, records, processor, augment=False):
        self.records   = records
        self.processor = processor
        self.augment   = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec       = self.records[idx]
        image     = Image.open(rec["image"]).convert("RGB")
        caption   = rec["caption"]
        instruction = rec.get("instruction",
            "Analyze this plant leaf image and provide a detailed diagnosis.")

        if self.augment:
            image = AUGMENT(image)

        # Build Qwen2.5-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": caption,
            },
        ]

        # Apply chat template (gives us text with <|im_start|> tokens)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process image + text together
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        input_ids      = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values   = inputs["pixel_values"]  # shape varies with image tokens

        # ── Label masking: only supervise the assistant response ─────────
        # Find where the assistant turn starts by encoding prompt-only
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": instruction},
                ],
            },
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.processor.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"].squeeze(0)
        prompt_len = min(len(prompt_ids), len(input_ids))

        labels = input_ids.clone()
        labels[:prompt_len] = -100                                    # mask prompt
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # mask pad

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,  # will need custom collation
            "labels":         labels,
        }


def collate_fn(batch):
    """Custom collator because pixel_values have variable shapes per sample."""
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels         = torch.stack([b["labels"]         for b in batch])
    # pixel_values: list of tensors, each [num_patches, C, H, W]
    pixel_values   = torch.cat([b["pixel_values"]    for b in batch], dim=0)
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "pixel_values":   pixel_values,
        "labels":         labels,
    }


def main():
    print("=" * 60)
    print("  Qwen2.5-VL-7B-Instruct — PlantVillage Fine-tuning")
    print("  4-bit QLoRA r64 | Flash Attention 2 | Effective BS=32")
    print("=" * 60)

    # ── 4-bit quantization config (NF4 + double quant) ───────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # nested quantization — saves ~0.4GB
    )

    # ── Load processor ────────────────────────────────────────────────────
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    # ── Load model with QLoRA + Flash Attention 2 ─────────────────────────
    print("Loading Qwen2.5-VL-7B in 4-bit with Flash Attention 2...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    print(f"Model on  : {next(model.parameters()).device}")
    print(f"VRAM avail: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Apply LoRA (r=64 on ALL linear layers) ────────────────────────────
    print("\nApplying QLoRA r64...")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",   # targets every nn.Linear in the model
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Load dataset ──────────────────────────────────────────────────────
    print("\nLoading captions.json...")
    with open(CAPTIONS) as f:
        data = json.load(f)

    # ── Pre-flight: verify label masking works ────────────────────────────
    print("\nRunning pre-flight checks...")
    check_ds  = PlantVillageDataset(data["train"][:1], processor, augment=False)
    s         = check_ds[0]
    supervised = (s["labels"] != -100).sum().item()
    total      = s["labels"].shape[0]
    print(f"  Supervised tokens : {supervised} / {total}")
    if supervised < 5:
        print("  ERROR: Label masking broken — all labels are masked!")
        return
    print("  Pre-flight OK.\n")

    train_ds = PlantVillageDataset(data["train"], processor, augment=True)
    val_ds   = PlantVillageDataset(data["val"],   processor, augment=False)
    print(f"Train : {len(train_ds)} | Val : {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    # Use paged AdamW for QLoRA — keeps optimizer states in CPU RAM
    from bitsandbytes.optim import PagedAdamW8bit
    optimizer    = PagedAdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"\nTotal optimizer steps : {total_steps}")
    print(f"Warmup steps          : {warmup_steps}")
    print(f"Effective batch size  : {BATCH_SIZE * GRAD_ACCUM}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    best_val_loss    = float("inf")
    patience_counter = 0
    global_step      = 0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            # Move all tensors to GPU
            gpu_batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**gpu_batch)
                loss    = outputs.loss / GRAD_ACCUM

            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOG_STEPS == 0:
                    avg_loss  = running_loss / (LOG_STEPS * GRAD_ACCUM)
                    vram_used = torch.cuda.memory_allocated() / 1e9
                    vram_peak = torch.cuda.max_memory_allocated() / 1e9
                    print(
                        f"Epoch {epoch+1}/{EPOCHS} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"VRAM: {vram_used:.1f}GB (peak {vram_peak:.1f}GB)"
                    )
                    running_loss = 0.0

                if global_step % SAVE_STEPS == 0:
                    ckpt = f"{OUTPUT_DIR}/checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    print(f"  Checkpoint saved → {ckpt}")

        # ── Epoch validation ──────────────────────────────────────────────
        print(f"\nValidating epoch {epoch + 1}...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                gpu_batch = {k: v.to("cuda") for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    val_loss += model(**gpu_batch).loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1} complete | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            model.save_pretrained(f"{OUTPUT_DIR}/best")
            processor.save_pretrained(f"{OUTPUT_DIR}/best")
            print(f"  Best model saved (val_loss={val_loss:.4f})\n")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}\n")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining complete!")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Saved to      : {OUTPUT_DIR}/best")


if __name__ == "__main__":
    main()

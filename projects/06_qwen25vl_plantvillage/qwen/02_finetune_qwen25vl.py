# 02_finetune_qwen25vl.py
# Fine-tunes Qwen2.5-VL-7B-Instruct on PlantVillage dataset.
# Fix: label masking uses the assistant token marker directly in input_ids,
#      not a separate tokenizer call, so image tokens are correctly accounted for.

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
import torchvision.transforms as T

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-VL-7B-Instruct"
CAPTIONS     = "captions.json"
OUTPUT_DIR   = "outputs/qwen25vl_plantvillage"

EPOCHS       = 3
BATCH_SIZE   = 2
GRAD_ACCUM   = 8           # effective batch = 16
LR           = 1e-4
MAX_PIXELS   = 512 * 28 * 28
MIN_PIXELS   = 256 * 28 * 28

LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05

SAVE_STEPS   = 300
LOG_STEPS    = 20
PATIENCE     = 3
# ─────────────────────────────────────────────────────────────────────────

AUGMENT = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.08),
    T.RandomResizedCrop(size=(448, 448), scale=(0.75, 1.0)),
    T.RandomGrayscale(p=0.05),
])

# Qwen2.5-VL marks the start of the assistant turn with this token sequence.
# We find this in input_ids to get the exact cutoff including image tokens.
ASSISTANT_MARKER = "<|im_start|>assistant"


class PlantVillageDataset(Dataset):
    def __init__(self, records, processor, augment=False):
        self.records        = records
        self.processor      = processor
        self.augment        = augment
        # Encode the assistant marker once to get its token ids
        self.marker_ids     = processor.tokenizer.encode(
            ASSISTANT_MARKER, add_special_tokens=False
        )
        self.marker_len     = len(self.marker_ids)
        self.marker_tensor  = torch.tensor(self.marker_ids)

    def __len__(self):
        return len(self.records)

    def _find_assistant_start(self, input_ids: torch.Tensor) -> int:
        """
        Find the position right AFTER the <|im_start|>assistant token sequence.
        This correctly accounts for image tokens inserted before the marker.
        Returns the index of the first caption token.
        """
        marker = self.marker_tensor
        n      = len(input_ids)
        m      = self.marker_len
        for i in range(n - m):
            if torch.equal(input_ids[i:i + m], marker):
                # +m skips the marker itself, +1 skips the newline after it
                return i + m + 1
        # Fallback: if marker not found, supervise everything
        return 0

    def __getitem__(self, idx):
        rec         = self.records[idx]
        image       = Image.open(rec["image"]).convert("RGB")
        caption     = rec["caption"]
        instruction = rec.get(
            "instruction",
            "Analyze this plant leaf image and provide a detailed diagnosis."
        )

        if self.augment:
            image = AUGMENT(image)

        # Full conversation: user + assistant
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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        enc = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        pixel_values   = enc["pixel_values"]
        image_grid_thw = enc["image_grid_thw"]

        # Find exact assistant start position in the full input_ids
        # (which includes image tokens) and mask everything before it
        assistant_start = self._find_assistant_start(input_ids)

        labels = input_ids.clone()
        labels[:assistant_start] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels":         labels,
        }


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id  = 0

    input_ids_list, attn_mask_list, labels_list = [], [], []
    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        input_ids_list.append(F.pad(b["input_ids"],      (0, pad_len), value=pad_id))
        attn_mask_list.append(F.pad(b["attention_mask"], (0, pad_len), value=0))
        labels_list.append(   F.pad(b["labels"],         (0, pad_len), value=-100))

    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_mask_list),
        "pixel_values":   torch.cat([b["pixel_values"]   for b in batch], dim=0),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], dim=0),
        "labels":         torch.stack(labels_list),
    }


def main():
    print("=" * 60)
    print("  Qwen2.5-VL-7B-Instruct — PlantVillage Fine-tuning")
    print("  bfloat16 | LoRA r32 | eff-BS 16 | marker-based label mask")
    print("=" * 60)

    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print("Loading Qwen2.5-VL-7B in bfloat16...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    print(f"Model on  : {next(model.parameters()).device}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print("\nApplying LoRA r32...")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("\nLoading captions.json...")
    with open(CAPTIONS) as f:
        data = json.load(f)

    print("\nPre-flight check...")
    check_ds   = PlantVillageDataset(data["train"][:1], processor, augment=False)
    s          = check_ds[0]
    supervised = (s["labels"] != -100).sum().item()
    total_toks = s["labels"].shape[0]
    print(f"  Total tokens       : {total_toks}")
    print(f"  Supervised tokens  : {supervised}  (caption tokens only)")
    print(f"  Masked tokens      : {total_toks - supervised}  (prompt + image)")
    print(f"  image_grid_thw     : {s['image_grid_thw'].shape}")
    print(f"  pixel_values shape : {s['pixel_values'].shape}")
    if supervised < 10:
        print("  ERROR: Too few supervised tokens — label masking broken!")
        return
    if supervised > total_toks * 0.8:
        print("  WARNING: Most tokens supervised — prompt may not be masked.")
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

    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
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
                        f"VRAM: {vram_used:.1f} GB (peak {vram_peak:.1f} GB)"
                    )
                    running_loss = 0.0

                if global_step % SAVE_STEPS == 0:
                    ckpt = f"{OUTPUT_DIR}/checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    print(f"  Checkpoint → {ckpt}")

        print(f"\nValidating epoch {epoch + 1}...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                gpu_batch = {k: v.to("cuda") for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    val_loss += model(**gpu_batch).loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            model.save_pretrained(f"{OUTPUT_DIR}/best")
            processor.save_pretrained(f"{OUTPUT_DIR}/best")
            print(f"  Best model saved (val_loss={val_loss:.4f})\n")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience {patience_counter}/{PATIENCE}\n")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining complete!")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Adapter saved : {OUTPUT_DIR}/best")


if __name__ == "__main__":
    main()

import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
import torchvision.transforms as T

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_NAME  = "Salesforce/blip2-opt-2.7b"
CAPTIONS    = "captions.json"
OUTPUT_DIR  = "outputs/blip2_plantvillage"
EPOCHS      = 5
BATCH_SIZE  = 8
GRAD_ACCUM  = 2          # effective batch size = 16
LR          = 1e-4
MAX_LENGTH  = 160
LORA_R      = 32
LORA_ALPHA  = 64
LORA_DROP   = 0.05
SAVE_STEPS  = 200
LOG_STEPS   = 20
PATIENCE    = 3          # early stopping patience (epochs)
# ─────────────────────────────────────────────────────────────────────────────

# ── Image augmentation (training only) ───────────────────────────────────────
AUGMENT = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
])


class PlantVillageDataset(Dataset):
    def __init__(self, records: list, processor: Blip2Processor, augment: bool = False):
        self.records   = records
        self.processor = processor
        self.augment   = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec     = self.records[idx]
        image   = Image.open(rec["image"]).convert("RGB")
        caption = rec["caption"]

        if self.augment:
            image = AUGMENT(image)

        enc = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        pixel_values   = enc["pixel_values"].squeeze(0)

        # ── Correct label masking: encode prompt-only to find exact cutoff ──
        # Mask out padding tokens so loss is only computed on caption tokens.
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "labels":         labels,
        }


def collate_fn(batch: list) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def main():
    print("=" * 60)
    print("  BLIP-2 OPT-2.7B — PlantVillage Fine-tuning")
    print(f"  LoRA r={LORA_R} | LR={LR} | Epochs={EPOCHS} | BS={BATCH_SIZE}x{GRAD_ACCUM}")
    print("=" * 60)

    # ── Processor ─────────────────────────────────────────────────────────────
    print("\nLoading processor...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME)

    # ── Base model in bfloat16 ────────────────────────────────────────────────
    print("Loading model in bfloat16...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Device : {next(model.parameters()).device}")
    print(f"  VRAM   : {total_vram:.1f} GB available")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    print("\nApplying LoRA...")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
        lora_dropout=LORA_DROP,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    with open(CAPTIONS) as f:
        data = json.load(f)

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    print("\nRunning pre-flight checks...")
    check_ds   = PlantVillageDataset(data["train"][:1], processor, augment=False)
    sample     = check_ds[0]
    supervised = (sample["labels"] != -100).sum().item()
    print(f"  Supervised tokens : {supervised} / {sample['labels'].shape[0]}")
    if supervised < 5:
        print("  ERROR: Label masking broken — no supervised tokens found.")
        return

    lb_caps = [r["caption"] for r in data["train"]
               if "late_blight" in r["class_name"].lower()][:10]
    unique  = len(set(lb_caps))
    print(f"  Caption variation : {unique}/10 unique in Late Blight sample")
    if unique < 5:
        print("  WARNING: Low variation — re-run 01_prepare_dataset.py")
    print("  Pre-flight OK.\n")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = PlantVillageDataset(data["train"], processor, augment=True)
    val_ds   = PlantVillageDataset(data["val"],   processor, augment=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * 0.05)
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
        # ── Training loop ─────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
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
                    lr_now    = scheduler.get_last_lr()[0]
                    print(
                        f"Ep {epoch+1}/{EPOCHS} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr_now:.2e} | "
                        f"VRAM: {vram_used:.1f}GB"
                    )
                    running_loss = 0.0

                if global_step % SAVE_STEPS == 0:
                    ckpt = f"{OUTPUT_DIR}/checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    print(f"  Checkpoint → {ckpt}")

        # ── Validation ────────────────────────────────────────────────────────
        print(f"\nValidating epoch {epoch + 1}...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    val_loss += model(**batch).loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            model.save_pretrained(f"{OUTPUT_DIR}/best")
            processor.save_pretrained(f"{OUTPUT_DIR}/best")
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})\n")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience {patience_counter}/{PATIENCE}\n")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Model saved   : {OUTPUT_DIR}/best")
    print("=" * 60)


if __name__ == "__main__":
    main()
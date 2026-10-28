# 02b_finetune_florence2.py
# Second VLM: Microsoft Florence-2-base fine-tuned on PlantVillage
# Compare against BLIP-2 OPT-2.7B from 02_finetune_blip2.py
#
# WHY FLORENCE-2?
#   - Sequence-to-sequence architecture (encoder-decoder), lighter than decoder-only OPT
#   - Native support for image captioning, OCR, grounding — versatile backbone
#   - 232M params (base) → trains fast on RTX 4500 Ada
#   - Good baseline for ablation: transformer VLM vs autoregressive VLM

import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import torchvision.transforms as T

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "microsoft/Florence-2-base-ft"    # base fine-tune checkpoint
CAPTIONS    = "captions.json"
OUTPUT_DIR  = "outputs/florence2_plantvillage"
EPOCHS      = 5
BATCH_SIZE  = 8
GRAD_ACCUM  = 2
LR          = 2e-4
MAX_LENGTH  = 160
LORA_R      = 16           # Florence-2 is smaller so r=16 is plenty
LORA_ALPHA  = 32
SAVE_STEPS  = 200
LOG_STEPS   = 20
PATIENCE    = 3
TASK_TOKEN  = "<CAPTION>"  # Florence-2 uses task tokens
# ─────────────────────────────────────────────────────────────────────────────

AUGMENT = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
])


class PlantVillageDataset(Dataset):
    def __init__(self, records, processor, augment=False):
        self.records   = records
        self.processor = processor
        self.augment   = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec    = self.records[idx]
        image  = Image.open(rec["image"]).convert("RGB")
        caption = rec["caption"]

        if self.augment:
            image = AUGMENT(image)

        # Florence-2 takes task token as text prompt
        enc = self.processor(
            text=TASK_TOKEN,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        # Encode the target caption separately for labels
        label_enc = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        pixel_values   = enc["pixel_values"].squeeze(0)
        labels         = label_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "labels":         labels,
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def main():
    print("=" * 60)
    print("  Florence-2-base — PlantVillage Fine-tuning")
    print("  Comparison model vs BLIP-2 OPT-2.7b")
    print("  LoRA r=16 | 10 caption templates | Augmentation")
    print("=" * 60)

    # ── Load processor & model ───────────────────────────────
    print("\nLoading Florence-2 processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("Loading Florence-2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params : {total_params:.1f}M")
    print(f"Model on     : {next(model.parameters()).device}")
    print(f"VRAM total   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Apply LoRA ───────────────────────────────────────────
    print("\nApplying LoRA r=16...")
    # Florence-2 language model uses attention projections
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Load data ────────────────────────────────────────────
    print("\nLoading dataset...")
    with open(CAPTIONS) as f:
        data = json.load(f)

    # Pre-flight check
    print("\nRunning pre-flight checks...")
    check_ds   = PlantVillageDataset(data["train"][:1], processor, augment=False)
    s          = check_ds[0]
    supervised = (s["labels"] != -100).sum().item()
    print(f"  Supervised tokens : {supervised} / {s['labels'].shape[0]}")
    if supervised < 5:
        print("  ERROR: Label masking broken — re-check captions.json")
        return
    print("  Pre-flight OK.\n")

    train_ds = PlantVillageDataset(data["train"], processor, augment=True)
    val_ds   = PlantVillageDataset(data["val"],   processor, augment=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    # ── Optimizer + scheduler ────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * 0.05)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"\nTotal steps         : {total_steps}")
    print(f"Warmup steps        : {warmup_steps}")
    print(f"Effective batch     : {BATCH_SIZE * GRAD_ACCUM}")

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
                    print(
                        f"Epoch {epoch+1}/{EPOCHS} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"VRAM: {vram_used:.1f}GB"
                    )
                    running_loss = 0.0

                if global_step % SAVE_STEPS == 0:
                    ckpt = f"{OUTPUT_DIR}/checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    print(f"  Checkpoint saved → {ckpt}")

        # ── Validation ───────────────────────────────────────
        print(f"\nRunning validation for epoch {epoch+1}...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    val_loss += model(**batch).loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} complete | Val loss: {val_loss:.4f}")

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
                print(f"Best val loss: {best_val_loss:.4f}")
                break

    print("\nFlorence-2 training complete!")
    print(f"Adapter saved to: {OUTPUT_DIR}/best")
    print("\nTo compare with BLIP-2, run:")
    print("  python 04_evaluate.py --model blip2")
    print("  python 04_evaluate.py --model florence2")


if __name__ == "__main__":
    main()
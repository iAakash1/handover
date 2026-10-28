# quick_train_blip2.py
# Place in: C:\Users\Admin\Desktop\VLM\tp\
#
# Run:
#   python quick_train_blip2.py
#
# Resume after interruption:
#   python quick_train_blip2.py --resume outputs/blip2_plantvillage/checkpoint-200
#
# Saves every 100 steps to outputs/blip2_plantvillage/checkpoint-NNN/
# Each checkpoint includes adapter weights + training_state.pt
# (optimizer, scheduler, step, epoch, best_val_loss)

import json, torch, argparse
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
import torchvision.transforms as T

BASE_MODEL   = "Salesforce/blip2-opt-2.7b"
CAPTIONS     = "captions.json"
OUTPUT_DIR   = "outputs/blip2_plantvillage"
EPOCHS       = 1
BATCH_SIZE   = 8
GRAD_ACCUM   = 2
LR           = 1e-4
MAX_LENGTH   = 160
LORA_R       = 32
LORA_ALPHA   = 64
SAVE_STEPS   = 100
LOG_STEPS    = 20

AUGMENT = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
])


class PlantDS(Dataset):
    def __init__(self, records, processor, augment=False):
        self.records   = records
        self.processor = processor
        self.augment   = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        image = Image.open(rec["image"]).convert("RGB")
        if self.augment:
            image = T.ToPILImage()(AUGMENT(T.ToTensor()(image)))
        enc = self.processor(
            images=image, text=rec["caption"],
            return_tensors="pt", padding="max_length",
            max_length=MAX_LENGTH, truncation=True,
        )
        ids    = enc["input_ids"].squeeze(0)
        labels = ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      ids,
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   enc["pixel_values"].squeeze(0),
            "labels":         labels,
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def save_checkpoint(model, processor, optimizer, scheduler, step, epoch,
                    best_val_loss, out_dir):
    """Save adapter + full training state so we can resume exactly."""
    ckpt_dir = Path(out_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    processor.save_pretrained(str(ckpt_dir))
    torch.save({
        "global_step":    step,
        "epoch":          epoch,
        "best_val_loss":  best_val_loss,
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
    }, str(ckpt_dir / "training_state.pt"))
    print(f"  Checkpoint saved -> {ckpt_dir}")


def load_checkpoint(resume_path, base_model, lora_cfg, processor,
                    optimizer, scheduler):
    """Load adapter weights + training state from a checkpoint dir."""
    resume_path = Path(resume_path)
    state_file  = resume_path / "training_state.pt"
    if not resume_path.exists():
        print(f"ERROR: Resume path not found: {resume_path}")
        return None, None, 0, 0, float("inf")

    print(f"Resuming from: {resume_path}")
    model = PeftModel.from_pretrained(base_model, str(resume_path))

    if state_file.exists():
        state = torch.load(str(state_file), map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step   = state["global_step"]
        start_epoch   = state["epoch"]
        best_val_loss = state["best_val_loss"]
        print(f"  Resumed at step={global_step}  epoch={start_epoch}"
              f"  best_val_loss={best_val_loss:.4f}")
    else:
        global_step   = 0
        start_epoch   = 0
        best_val_loss = float("inf")
        print("  No training_state.pt found — starting counters from zero.")

    return model, optimizer, global_step, start_epoch, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint dir to resume from "
                             "(e.g. outputs/blip2_plantvillage/checkpoint-200)")
    args = parser.parse_args()

    print("=" * 55)
    print("  BLIP-2 OPT-2.7B  |  PlantVillage  |  LoRA r32")
    if args.resume:
        print(f"  Mode: RESUME from {args.resume}")
    else:
        print("  Mode: FRESH training (1 epoch)")
    print("=" * 55)

    if not Path(CAPTIONS).exists():
        print("ERROR: captions.json not found. Run 01_prepare_dataset.py first.")
        return

    with open(CAPTIONS) as f:
        data = json.load(f)

    train_records = data["train"] if "train" in data else data
    val_records   = data.get("val", train_records[:int(len(train_records) * 0.1)])
    print(f"Train: {len(train_records)}  |  Val: {len(val_records)}")

    print("\nLoading processor...")
    proc_source = args.resume if args.resume else BASE_MODEL
    processor   = Blip2Processor.from_pretrained(proc_source)

    print("Loading base model in bfloat16...")
    base = Blip2ForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base.gradient_checkpointing_enable()

    total_steps  = (len(train_records) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
    warmup_steps = int(total_steps * 0.05)

    optimizer = torch.optim.AdamW(
        base.parameters(), lr=LR, weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if args.resume:
        model, optimizer, global_step, start_epoch, best_val_loss = load_checkpoint(
            args.resume, base, None, processor, optimizer, scheduler
        )
        if model is None:
            return
    else:
        print("Applying LoRA...")
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj","k_proj","v_proj","o_proj","fc1","fc2"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model         = get_peft_model(base, lora_cfg)
        global_step   = 0
        start_epoch   = 0
        best_val_loss = float("inf")

    model.print_trainable_parameters()

    train_ds = PlantDS(train_records, processor, augment=True)
    val_ds   = PlantDS(val_records,   processor, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"\nTotal optimizer steps : {total_steps}")
    print(f"Warmup steps          : {warmup_steps}")
    print(f"Save every            : {SAVE_STEPS} steps")
    print("\nStarting training...\n")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / GRAD_ACCUM
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOG_STEPS == 0:
                    avg  = running_loss / (LOG_STEPS * GRAD_ACCUM)
                    vram = torch.cuda.memory_allocated() / 1e9
                    pk   = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                          f"Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                          f"VRAM: {vram:.1f}/{pk:.1f}GB")
                    running_loss = 0.0

                if global_step % SAVE_STEPS == 0:
                    save_checkpoint(model, processor, optimizer, scheduler,
                                    global_step, epoch, best_val_loss, OUTPUT_DIR)

        print(f"\nValidating epoch {epoch+1}...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    val_loss += model(**batch).loss.item()
        val_loss /= len(val_loader)
        print(f"  Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = Path(OUTPUT_DIR) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            processor.save_pretrained(str(best_dir))
            print(f"  Best model saved -> {best_dir}\n")

    print("=" * 55)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved -> {OUTPUT_DIR}/best/")
    print("\nNow run: python quick_eval_blip2.py")
    print("=" * 55)


if __name__ == "__main__":
    main()

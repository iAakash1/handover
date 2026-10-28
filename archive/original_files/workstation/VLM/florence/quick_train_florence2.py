# quick_train_florence2.py
# Place in: C:\Users\Admin\Desktop\VLM\florence\
#
# Run patch first:
#   python patch_florence2.py
#
# Then train:
#   python quick_train_florence2.py
#
# Resume:
#   python quick_train_florence2.py --resume outputs/florence2_plantvillage/checkpoint-200

import json, torch, argparse
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
import torchvision.transforms as T

MODEL_NAME  = "microsoft/Florence-2-base-ft"
CAPTIONS    = "../tp/captions.json"     # use clean captions from tp/
OUTPUT_DIR  = "outputs/florence2_plantvillage"
EPOCHS      = 3
BATCH_SIZE  = 8
GRAD_ACCUM  = 2
LR          = 2e-4
MAX_LENGTH  = 160
LORA_R      = 16
LORA_ALPHA  = 32
SAVE_STEPS  = 100
LOG_STEPS   = 20
TASK_TOKEN  = "<CAPTION>"

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
            image = AUGMENT(image)

        enc = self.processor(
            text=TASK_TOKEN, images=image,
            return_tensors="pt", padding="max_length",
            max_length=MAX_LENGTH, truncation=True,
        )
        label_enc = self.processor.tokenizer(
            rec["caption"], return_tensors="pt",
            padding="max_length", max_length=MAX_LENGTH, truncation=True,
        )
        labels = label_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   enc["pixel_values"].squeeze(0),
            "labels":         labels,
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def build_model_fresh(base):
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj","k_proj","v_proj","out_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model


def build_optimizer(model, total_steps, warmup_steps):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def save_checkpoint(model, processor, optimizer, scheduler,
                    step, epoch, best_val_loss):
    ckpt = Path(OUTPUT_DIR) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    processor.save_pretrained(str(ckpt))
    torch.save({
        "global_step":   step,
        "epoch":         epoch,
        "best_val_loss": best_val_loss,
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
    }, str(ckpt / "training_state.pt"))
    print(f"  Checkpoint saved -> {ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    print("=" * 55)
    print("  Florence-2-base-ft  |  PlantVillage  |  LoRA r16")
    print(f"  Mode: {'RESUME from ' + args.resume if args.resume else 'FRESH training'}")
    print("=" * 55)

    if not Path(CAPTIONS).exists():
        print(f"ERROR: captions.json not found at {CAPTIONS}")
        print("Make sure tp/captions.json exists.")
        return

    with open(CAPTIONS) as f:
        data = json.load(f)

    train_records = data["train"] if "train" in data else data
    val_records   = data.get("val", train_records[:int(len(train_records)*0.1)])
    print(f"Train: {len(train_records)}  |  Val: {len(val_records)}")

    total_steps  = (len(train_records) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
    warmup_steps = int(total_steps * 0.05)

    print("\nLoading Florence-2 base model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False,  # disable meta tensors — DaViT.__init__ calls .item()
    ).to("cuda")
    base.gradient_checkpointing_enable()
    total_params = sum(p.numel() for p in base.parameters()) / 1e6
    print("  Params: {:.1f}M  |  VRAM: {:.1f}GB".format(
        total_params, torch.cuda.get_device_properties(0).total_memory/1e9))

    if args.resume:
        resume_path = Path(args.resume)
        state_file  = resume_path / "training_state.pt"
        print(f"Resuming from: {resume_path}")

        model = PeftModel.from_pretrained(base, str(resume_path))
        optimizer, scheduler = build_optimizer(model, total_steps, warmup_steps)

        if state_file.exists():
            state = torch.load(str(state_file), map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step   = state["global_step"]
            start_epoch   = state["epoch"]
            best_val_loss = state["best_val_loss"]
            print(f"  Resumed: step={global_step}  epoch={start_epoch}  best_val_loss={best_val_loss:.4f}")
        else:
            global_step   = 0
            start_epoch   = 0
            best_val_loss = float("inf")

        processor = AutoProcessor.from_pretrained(str(resume_path), trust_remote_code=True)
    else:
        print("Applying LoRA r=16...")
        model         = build_model_fresh(base)
        optimizer, scheduler = build_optimizer(model, total_steps, warmup_steps)
        global_step   = 0
        start_epoch   = 0
        best_val_loss = float("inf")
        processor     = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    train_ds = PlantDS(train_records, processor, augment=True)
    val_ds   = PlantDS(val_records,   processor, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\nTotal steps  : {total_steps}")
    print(f"Warmup steps : {warmup_steps}")
    print(f"Checkpoints  : every {SAVE_STEPS} steps")
    print("\nStarting training...\n")

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
                    print(f"  Epoch {epoch+1}/{EPOCHS} | Step {global_step}/{total_steps} | "
                          f"Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                          f"VRAM: {vram:.1f}/{pk:.1f}GB")
                    running_loss = 0.0

                if global_step % SAVE_STEPS == 0:
                    save_checkpoint(model, processor, optimizer, scheduler,
                                    global_step, epoch, best_val_loss)

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
        else:
            print(f"  No improvement over {best_val_loss:.4f}\n")

    print("=" * 55)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print("Now run: python quick_eval_florence2.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
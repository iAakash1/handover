# LLaVA-1.6-Mistral-7B + LoRA on the PlantVillage captions (March 2026 experiment, rebuilt).
#
# The original March training script was not kept: only its adapter survived
# (outputs/llava_plantvillage/best + checkpoint-100 ... checkpoint-1700). This script rebuilds that run from
# what the saved adapter records:
#   base model llava-hf/llava-v1.6-mistral-7b-hf, LoRA r=64, alpha=128, dropout=0.05 on
#   q/k/v/o/gate/up/down projections, adapter saved every 100 steps, best adapter kept on validation loss.
# Not recorded anywhere, so taken from our other VLM runs: LR 2e-4, batch 1 x grad-accum 8, bf16.
# Batch size stays 1 on purpose: LLaVA-1.6 cuts each photo into a different number of tiles, and batching
# those is what broke our LLaVA-1.5 attempt (batch-collation error).
#
# Run from this folder with the vlm env (24 GB GPU):
#   ..\..\.venvs\vlm\Scripts\python 02_finetune_llava.py                       # full run, ~36 s per step
#   ..\..\.venvs\vlm\Scripts\python 02_finetune_llava.py --max-steps 2 --val 2  # quick check that it trains
# It writes to outputs/llava_plantvillage_rerun so the March adapter is never overwritten.

import argparse
import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (LlavaNextForConditionalGeneration, LlavaNextProcessor,
                          get_cosine_schedule_with_warmup)

HERE = Path(__file__).resolve().parent
MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
CAPTIONS = HERE.parent.parent / "projects" / "06_qwen25vl_plantvillage" / "qwen" / "captions.json"   # same captions as every VLM
OLD_ROOT = r"C:\Users\Admin\Desktop\VLM\PlantVillage"                           # prefix stored in captions.json
DATA_ROOT = os.environ.get("PLANTVILLAGE_DIR", OLD_ROOT)

LR = 2e-4
GRAD_ACCUM = 8
LORA_R, LORA_ALPHA, LORA_DROP = 64, 128, 0.05
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SAVE_STEPS = 100
SEED = 42


def load_image(path):
    # captions.json was written on the TIH machine; swap the folder if the dataset lives elsewhere
    return Image.open(path.replace(OLD_ROOT, DATA_ROOT)).convert("RGB")


def encode(proc, rec):
    """Prompt + answer as one sequence; loss only on the answer tokens."""
    image = load_image(rec["image"])
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": rec["instruction"]}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    full = proc(text=prompt + " " + rec["caption"] + proc.tokenizer.eos_token, images=image, return_tensors="pt")
    n_prompt = proc(text=prompt, images=image, return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    full["labels"] = labels
    return {k: v.to("cuda") for k, v in full.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = no limit)")
    ap.add_argument("--val", type=int, default=50, help="validation images used for the best-adapter check")
    ap.add_argument("--out", default=str(HERE / "outputs" / "llava_plantvillage_rerun"))
    args = ap.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    data = json.load(open(CAPTIONS, encoding="utf-8"))
    train, val = data["train"], data["val"][:args.val]

    proc = LlavaNextProcessor.from_pretrained(MODEL_NAME)
    model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                              device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROP,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    total = args.max_steps or (len(train) * args.epochs) // GRAD_ACCUM
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)

    def val_loss():
        model.eval()
        with torch.no_grad():
            losses = [model(**encode(proc, r)).loss.item() for r in val]
        model.train()
        return sum(losses) / len(losses)

    out = Path(args.out)
    best, step, micro = float("inf"), 0, 0
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(train)
        for rec in train:
            loss = model(**encode(proc, rec)).loss / GRAD_ACCUM
            loss.backward()
            micro += 1
            if micro % GRAD_ACCUM:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0 or step == total:
                print(f"epoch {epoch + 1} step {step}/{total} loss {loss.item() * GRAD_ACCUM:.4f}", flush=True)
            if step % SAVE_STEPS == 0 or step == total:
                model.save_pretrained(out / f"checkpoint-{step}")
                vl = val_loss()
                print(f"step {step}: validation loss {vl:.4f}", flush=True)
                if vl < best:  # keep the adapter with the lowest validation loss as best/
                    best = vl
                    model.save_pretrained(out / "best")
                    proc.save_pretrained(out / "best")
            if step >= total:
                print(f"done, best validation loss {best:.4f} -> {out / 'best'}")
                return


if __name__ == "__main__":
    main()

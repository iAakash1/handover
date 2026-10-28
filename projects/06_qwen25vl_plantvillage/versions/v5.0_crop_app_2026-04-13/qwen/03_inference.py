# 03_inference.py
# Inference for fine-tuned Qwen2.5-VL-7B-Instruct on PlantVillage.
# Runs one image per class, applies CLIP OOD detection, saves results.
#
# Save to  : C:\Users\Admin\Desktop\VLM\qwen\
# Run      : python 03_inference.py

import torch
import json
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
)
from peft import PeftModel

# ── Paths ─────────────────────────────────────────────────────────────────
ADAPTER_PATH = "outputs/qwen25vl_plantvillage/best"
BASE_MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"
PLANT_DIR    = r"C:\Users\Admin\Desktop\VLM\PlantVillage"
OUTPUT_FILE  = "inference_results.json"

# ── Generation ────────────────────────────────────────────────────────────
INFERENCE_INSTRUCTION = (
    "Analyze this plant leaf image and provide a detailed diagnosis. "
    "Include: plant name, disease or health condition, severity level, "
    "causal pathogen, and visible symptoms."
)
MAX_NEW_TOKENS = 160
NUM_BEAMS      = 4

# ── OOD ───────────────────────────────────────────────────────────────────
CLIP_MODEL_NAME   = "openai/clip-vit-base-patch32"
OOD_THRESHOLD     = 0.22
PLANT_LEAF_PROMPT = "a photo of a plant leaf"


def load_model():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

    print("Loading Qwen2.5-VL-7B in bfloat16 with Flash Attention 2...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",

    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print("VLM ready.\n")
    return model, processor


def load_clip():
    print("Loading CLIP for OOD detection...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to("cuda")
    clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()
    return clip_model, clip_proc


def is_ood(image, clip_model, clip_proc, threshold=OOD_THRESHOLD):
    """Returns (is_ood: bool, similarity_score: float)."""
    inputs = clip_proc(
        text=[PLANT_LEAF_PROMPT],
        images=image,
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    with torch.no_grad():
        out     = clip_model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        sim     = (img_emb * txt_emb).sum().item()
    return sim < threshold, round(sim, 4)


def caption_image(image, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": INFERENCE_INSTRUCTION},
            ],
        }
    ]
    text   = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            repetition_penalty=2.5,
            no_repeat_ngram_size=6,
            length_penalty=0.8,
            early_stopping=True,
        )
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def main():
    model,      processor  = load_model()
    clip_model, clip_proc  = load_clip()

    test_path = Path(PLANT_DIR)
    results   = []

    print("Running inference — one image per class\n")
    print("=" * 70)

    for class_folder in sorted(test_path.iterdir()):
        if not class_folder.is_dir():
            continue
        images = (
            list(class_folder.glob("*.jpg"))
            + list(class_folder.glob("*.JPG"))
            + list(class_folder.glob("*.png"))
        )
        if not images:
            continue

        image = Image.open(images[0]).convert("RGB")

        ood_flag, ood_score = is_ood(image, clip_model, clip_proc)

        if ood_flag:
            caption = f"[OOD] Image does not appear to be a plant leaf (CLIP score: {ood_score:.3f})"
        else:
            caption = caption_image(image, model, processor)

        print(f"Class   : {class_folder.name}")
        print(f"OOD     : {ood_flag} (score: {ood_score:.3f})")
        print(f"Caption : {caption}")
        print("-" * 70)

        results.append({
            "image":     str(images[0]),
            "class":     class_folder.name,
            "caption":   caption,
            "ood":       ood_flag,
            "ood_score": ood_score,
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    in_dist = sum(1 for r in results if not r["ood"])
    print(f"\nDone. {len(results)} results → {OUTPUT_FILE}")
    print(f"In-distribution: {in_dist} | OOD flagged: {len(results) - in_dist}")


if __name__ == "__main__":
    main()

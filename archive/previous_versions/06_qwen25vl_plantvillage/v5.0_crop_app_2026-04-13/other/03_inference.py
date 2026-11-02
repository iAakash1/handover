# 03_inference.py
# Runs inference on PlantVillage using fine-tuned Qwen2.5-VL-7B-Instruct
# Includes CLIP-based Out-Of-Distribution (OOD) detection
# One image per class is sampled; results saved to inference_results.json

import torch
import json
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    CLIPProcessor,
    CLIPModel,
)
from peft import PeftModel

# ── Paths ─────────────────────────────────────────────────────────────────
ADAPTER_PATH = "outputs/qwen25vl_plantvillage/best"
PLANT_DIR    = r"C:\Users\Admin\Desktop\VLM\PlantVillage"
OUTPUT_FILE  = "inference_results.json"

# ── OOD Detection Config ──────────────────────────────────────────────────
CLIP_MODEL_NAME   = "openai/clip-vit-base-patch32"
OOD_THRESHOLD     = 0.22   # cosine similarity below this = out-of-distribution
PLANT_LEAF_PROMPT = "a photo of a plant leaf"

# ── Generation Config ─────────────────────────────────────────────────────
INFERENCE_INSTRUCTION = (
    "Analyze this plant leaf image and provide a detailed diagnosis. "
    "Include: plant name, disease or health condition, severity level, "
    "causal pathogen, and visible symptoms."
)
MAX_NEW_TOKENS = 160
NUM_BEAMS      = 4


def load_clip():
    """Load CLIP for OOD detection."""
    print("Loading CLIP for OOD detection...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to("cuda")
    clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()
    return clip_model, clip_proc


def is_ood(image: Image.Image, clip_model, clip_proc, threshold=OOD_THRESHOLD) -> tuple:
    """
    Returns (is_ood: bool, similarity_score: float).
    If cosine similarity to 'a photo of a plant leaf' is below threshold,
    the image is flagged as out-of-distribution.
    """
    inputs = clip_proc(
        text=[PLANT_LEAF_PROMPT],
        images=image,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    with torch.no_grad():
        outputs = clip_model(**inputs)
        img_emb  = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb  = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
        sim      = (img_emb * txt_emb).sum().item()

    return sim < threshold, round(sim, 4)


def load_model():
    """Load Qwen2.5-VL with 4-bit quantization and LoRA adapter."""
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

    print("Loading Qwen2.5-VL-7B in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print("Model ready.\n")
    return model, processor


def caption_image(image: Image.Image, model, processor) -> str:
    """Generate a structured plant disease caption for an image."""
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
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to("cuda")

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

    # Strip input tokens from output
    input_len = inputs["input_ids"].shape[1]
    generated = out[0][input_len:]
    raw       = processor.decode(generated, skip_special_tokens=True).strip()
    return raw


def main():
    model,     processor  = load_model()
    clip_model, clip_proc = load_clip()

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

        img_path = images[0]
        image    = Image.open(img_path).convert("RGB")

        # OOD check
        ood_flag, ood_score = is_ood(image, clip_model, clip_proc)

        if ood_flag:
            caption = f"[OOD] Image does not appear to be a plant leaf (CLIP score: {ood_score:.3f})"
        else:
            caption = caption_image(image, model, processor)

        print(f"Class   : {class_folder.name}")
        print(f"OOD     : {ood_flag} (CLIP score: {ood_score:.3f})")
        print(f"Caption : {caption}")
        print("-" * 70)

        results.append({
            "image":     str(img_path),
            "class":     class_folder.name,
            "caption":   caption,
            "ood":       ood_flag,
            "ood_score": ood_score,
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    in_dist = [r for r in results if not r["ood"]]
    print(f"\nDone. {len(results)} results saved to {OUTPUT_FILE}")
    print(f"In-distribution: {len(in_dist)} | OOD flagged: {len(results) - len(in_dist)}")


if __name__ == "__main__":
    main()

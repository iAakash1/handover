# 03_inference.py  (v2 — multi-model + OOD detection)
#
# NEW IN V2:
#   --model blip2 | florence2        select which fine-tuned VLM to run
#   --ood                            enable out-of-distribution detection
#   --threshold 0.45                 confidence gate below which image is flagged as OOD
#
# OOD DETECTION STRATEGY:
#   We parse the generated caption for the structured fields (Plant / Condition / Severity).
#   If the model cannot produce a valid structured caption — or if CLIP cosine similarity
#   between the image and the 15 known class text embeddings is below the threshold — the
#   image is flagged as "Out of Distribution: not a known plant leaf."
#   This gives two independent signals that must both agree before classifying a result.

import argparse
import json
import re
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    CLIPProcessor,
    CLIPModel,
)
from peft import PeftModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BLIP2_BASE      = "Salesforce/blip2-opt-2.7b"
BLIP2_ADAPTER   = "outputs/blip2_plantvillage/best"
FLORENCE2_BASE  = "microsoft/Florence-2-base-ft"
FLORENCE2_ADAPTER = "outputs/florence2_plantvillage/best"
CLIP_MODEL_ID   = "openai/clip-vit-base-patch32"
PLANT_DIR       = r"C:\Users\Admin\Desktop\VLM\PlantVillage"
RESULTS_FILE    = "inference_results.json"

# ── Known class labels for CLIP similarity ────────────────────────────────────
KNOWN_CLASS_DESCRIPTIONS = [
    "a tomato leaf with late blight disease showing dark lesions",
    "a tomato leaf with early blight disease showing target spots",
    "a tomato leaf with bacterial spot showing dark dots and yellow halo",
    "a tomato leaf with leaf mold showing olive fungal growth",
    "a tomato leaf with septoria leaf spot showing small round lesions",
    "a tomato leaf damaged by spider mites showing stippling",
    "a tomato leaf with target spot disease showing ringed brown spots",
    "a tomato leaf infected with mosaic virus showing mottled green pattern",
    "a tomato leaf with yellow leaf curl virus showing curling and yellowing",
    "a healthy tomato plant leaf with uniform green color",
    "a potato leaf with late blight disease",
    "a potato leaf with early blight disease",
    "a healthy potato leaf",
    "a pepper bell leaf with bacterial spot disease",
    "a healthy pepper bell leaf",
]

# ── Structured caption parser ─────────────────────────────────────────────────
STRUCTURED_PATTERN = re.compile(
    r"Plant:\s*(?P<plant>[^.]+)\.\s*"
    r"Condition:\s*(?P<condition>[^.]+)\.\s*"
    r"Severity:\s*(?P<severity>[^.]+)\.",
    re.IGNORECASE,
)

VALID_PLANTS     = {"tomato", "potato", "pepper bell", "pepper"}
VALID_CONDITIONS = {
    "late blight", "early blight", "bacterial spot", "leaf mold",
    "septoria leaf spot", "spider mite damage", "target spot",
    "mosaic virus", "tomato yellowleaf curl virus", "yellow leaf curl virus",
    "healthy",
}


def parse_caption(caption: str):
    """Return (plant, condition, severity) or None if caption is malformed."""
    m = STRUCTURED_PATTERN.search(caption)
    if not m:
        return None
    return m.group("plant").strip(), m.group("condition").strip(), m.group("severity").strip()


# ── CLIP-based OOD detector ───────────────────────────────────────────────────
class OODDetector:
    def __init__(self, threshold=0.45):
        print(f"Loading CLIP for OOD detection (threshold={threshold})...")
        self.clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to("cuda").half()
        self.clip_model.eval()
        self.threshold  = threshold

        # Pre-compute text embeddings for all known classes
        text_inputs = self.clip_proc(
            text=KNOWN_CLASS_DESCRIPTIONS,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        with torch.no_grad():
            self.text_embeds = self.clip_model.get_text_features(**text_inputs)
            self.text_embeds = self.text_embeds / self.text_embeds.norm(dim=-1, keepdim=True)
        print(f"  CLIP text embeddings cached for {len(KNOWN_CLASS_DESCRIPTIONS)} classes.")

    @torch.no_grad()
    def is_ood(self, image: Image.Image):
        """Returns (is_ood: bool, max_similarity: float, best_match: str)."""
        img_input = self.clip_proc(images=image, return_tensors="pt").to("cuda")
        img_embed = self.clip_model.get_image_features(**img_input)
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

        sims       = (img_embed @ self.text_embeds.T).squeeze(0).float().cpu().numpy()
        max_sim    = float(sims.max())
        best_idx   = int(sims.argmax())
        best_match = KNOWN_CLASS_DESCRIPTIONS[best_idx]

        return max_sim < self.threshold, max_sim, best_match


# ── Model loaders ─────────────────────────────────────────────────────────────
def load_blip2():
    print("Loading BLIP-2 processor + adapter...")
    processor = Blip2Processor.from_pretrained(BLIP2_ADAPTER)
    base      = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_BASE, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, BLIP2_ADAPTER)
    model.eval()
    return model, processor, "BLIP-2 OPT-2.7B"


def load_florence2():
    print("Loading Florence-2 processor + adapter...")
    processor = AutoProcessor.from_pretrained(FLORENCE2_ADAPTER, trust_remote_code=True)
    base      = AutoModelForCausalLM.from_pretrained(
        FLORENCE2_BASE,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, FLORENCE2_ADAPTER)
    model.eval()
    return model, processor, "Florence-2-base"


# ── Caption generators ────────────────────────────────────────────────────────
def caption_blip2(image: Image.Image, model, processor) -> str:
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.bfloat16)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,
            repetition_penalty=3.0,
            no_repeat_ngram_size=8,
            length_penalty=0.6,
            early_stopping=True,
        )
    raw       = processor.decode(out[0], skip_special_tokens=True).strip()
    sentences = raw.split(". ")
    clean     = ". ".join(sentences[:3]).strip()
    return clean if clean.endswith(".") else clean + "."


def caption_florence2(image: Image.Image, model, processor) -> str:
    task = "<CAPTION>"
    inputs = processor(text=task, images=image, return_tensors="pt").to("cuda", torch.bfloat16)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,
            repetition_penalty=2.5,
            no_repeat_ngram_size=6,
            length_penalty=0.6,
            early_stopping=True,
        )
    raw = processor.decode(out[0], skip_special_tokens=True).strip()
    # Florence-2 may echo the task token — remove it
    raw = raw.replace("<CAPTION>", "").strip()
    sentences = raw.split(". ")
    clean     = ". ".join(sentences[:3]).strip()
    return clean if clean.endswith(".") else clean + "."


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PlantVillage VLM Inference v2")
    parser.add_argument("--model",     choices=["blip2", "florence2"], default="blip2",
                        help="Which fine-tuned model to run")
    parser.add_argument("--ood",       action="store_true",
                        help="Enable CLIP-based OOD detection")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="CLIP cosine similarity threshold for OOD (default 0.45)")
    parser.add_argument("--image",     type=str, default=None,
                        help="Path to a single image (skips dataset scan)")
    args = parser.parse_args()

    # Load VLM
    if args.model == "blip2":
        model, processor, model_name = load_blip2()
        caption_fn = lambda img: caption_blip2(img, model, processor)
    else:
        model, processor, model_name = load_florence2()
        caption_fn = lambda img: caption_florence2(img, model, processor)

    # Load OOD detector
    ood_detector = OODDetector(threshold=args.threshold) if args.ood else None

    results = []

    # ── Single image mode ──────────────────────────────────────────────────────
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"ERROR: Image not found at {img_path}")
            return

        image   = Image.open(img_path).convert("RGB")
        caption = caption_fn(image)
        parsed  = parse_caption(caption)

        result = {
            "image":      str(img_path),
            "model":      model_name,
            "caption":    caption,
        }

        if parsed:
            plant, condition, severity = parsed
            result["plant"]     = plant
            result["condition"] = condition
            result["severity"]  = severity
            is_in_dist = (
                plant.lower() in VALID_PLANTS and
                condition.lower() in VALID_CONDITIONS
            )
            result["structured_valid"] = is_in_dist
        else:
            result["structured_valid"] = False

        if ood_detector:
            is_ood, sim, match = ood_detector.is_ood(image)
            result["clip_similarity"]  = round(sim, 4)
            result["clip_best_match"]  = match
            result["ood_detected"]     = is_ood or not result["structured_valid"]

        print("\n" + "=" * 65)
        print(f"  Model   : {model_name}")
        print(f"  Image   : {img_path.name}")
        print(f"  Caption : {caption}")
        if ood_detector:
            ood_flag = result.get("ood_detected", False)
            print(f"  OOD     : {'⚠ YES — not a known plant leaf' if ood_flag else '✓ In-distribution'}")
            print(f"  CLIP sim: {result.get('clip_similarity', 'N/A'):.4f}")
        print("=" * 65)

        with open("single_inference.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved → single_inference.json")
        return

    # ── Dataset scan mode ─────────────────────────────────────────────────────
    test_path = Path(PLANT_DIR)
    print(f"\nRunning inference on PlantVillage — one image per class\n{'='*65}")

    for class_folder in sorted(test_path.iterdir()):
        if not class_folder.is_dir():
            continue
        images = (list(class_folder.glob("*.jpg")) +
                  list(class_folder.glob("*.JPG")) +
                  list(class_folder.glob("*.png")))
        if not images:
            continue

        image   = Image.open(str(images[0])).convert("RGB")
        caption = caption_fn(image)
        parsed  = parse_caption(caption)

        entry = {
            "image":   str(images[0]),
            "class":   class_folder.name,
            "model":   model_name,
            "caption": caption,
        }

        if parsed:
            plant, condition, severity = parsed
            entry["plant"]     = plant
            entry["condition"] = condition
            entry["severity"]  = severity

        if ood_detector:
            is_ood, sim, match = ood_detector.is_ood(image)
            entry["clip_similarity"] = round(sim, 4)
            entry["ood_detected"]    = is_ood

        print(f"Class  : {class_folder.name}")
        print(f"Caption: {caption}")
        if ood_detector:
            flag = entry.get("ood_detected", False)
            print(f"OOD    : {'⚠ YES' if flag else '✓ No'} | CLIP sim: {entry.get('clip_similarity', 'N/A'):.4f}")
        print("-" * 65)
        results.append(entry)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {len(results)} captions saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
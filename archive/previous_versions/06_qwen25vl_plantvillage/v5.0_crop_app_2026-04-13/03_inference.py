import json
import torch
from pathlib import Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Salesforce/blip2-opt-2.7b"
ADAPTER_PATH = "outputs/blip2_plantvillage/best"
PLANT_DIR    = r"C:\Users\Admin\Desktop\VLM\PlantVillage"
OUTPUT_JSON  = "inference_results.json"

# ── Generation config ─────────────────────────────────────────────────────────
GEN_KWARGS = dict(
    max_new_tokens       = 80,
    num_beams            = 4,
    repetition_penalty   = 3.0,
    no_repeat_ngram_size = 8,
    length_penalty       = 0.6,
    early_stopping       = True,
)


def load_model():
    """Load base BLIP-2 + LoRA adapter, return (model, processor)."""
    print("Loading processor...")
    processor = Blip2Processor.from_pretrained(ADAPTER_PATH)

    print("Loading base model (blip2-opt-2.7b) in bfloat16...")
    base = Blip2ForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print("Model ready.\n")
    return model, processor


def caption_image(image_path: str, model, processor) -> str:
    """Generate a caption for a single image."""
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.bfloat16)

    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS)

    raw = processor.decode(out[0], skip_special_tokens=True).strip()

    # Keep first 3 sentences to stay concise
    sentences = raw.split(". ")
    clean     = ". ".join(sentences[:3]).strip()
    if not clean.endswith("."):
        clean += "."
    return clean


def main():
    model, processor = load_model()
    test_path        = Path(PLANT_DIR)
    results          = []

    print("Running inference — one image per class\n")
    print("=" * 65)

    for class_folder in sorted(test_path.iterdir()):
        if not class_folder.is_dir():
            continue

        images = (
            list(class_folder.glob("*.jpg")) +
            list(class_folder.glob("*.JPG")) +
            list(class_folder.glob("*.jpeg")) +
            list(class_folder.glob("*.png"))
        )
        if not images:
            print(f"  SKIP (no images): {class_folder.name}")
            continue

        img_path = str(images[0])
        caption  = caption_image(img_path, model, processor)

        print(f"Class  : {class_folder.name}")
        print(f"Caption: {caption}")
        print("-" * 65)

        results.append({
            "image":   img_path,
            "class":   class_folder.name,
            "caption": caption,
        })

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} captions saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
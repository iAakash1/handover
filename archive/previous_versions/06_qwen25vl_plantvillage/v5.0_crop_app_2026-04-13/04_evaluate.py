import json
import torch
import nltk
from pathlib import Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_module

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Salesforce/blip2-opt-2.7b"      # must match training
ADAPTER_PATH = "outputs/blip2_plantvillage/best"
CAPTIONS     = "captions.json"
OUTPUT_JSON  = "eval_results.json"
MAX_EVAL     = 200   # set to None to evaluate entire val split

# ── Generation config (same as inference) ─────────────────────────────────────
GEN_KWARGS = dict(
    max_new_tokens       = 80,
    num_beams            = 4,
    repetition_penalty   = 3.0,
    no_repeat_ngram_size = 8,
    length_penalty       = 0.6,
    early_stopping       = True,
)


def load_model():
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


def generate_caption(image_path: str, model, processor) -> str:
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def main():
    # ── Load validation data ──────────────────────────────────────────────────
    with open(CAPTIONS) as f:
        val_data = json.load(f)["val"]

    if MAX_EVAL is not None:
        val_data = val_data[:MAX_EVAL]

    print(f"Evaluating on {len(val_data)} validation images...\n")

    model, processor = load_model()

    # ── Evaluation loop ───────────────────────────────────────────────────────
    rouge_scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    refs         = []
    hyps         = []
    rouge_scores = []
    per_sample   = []

    for i, rec in enumerate(val_data):
        pred = generate_caption(rec["image"], model, processor)
        ref  = rec["caption"]

        refs.append([ref.lower().split()])
        hyps.append(pred.lower().split())
        r_score = rouge_scorer.score(ref, pred)["rougeL"].fmeasure
        rouge_scores.append(r_score)

        per_sample.append({
            "image":     rec["image"],
            "class":     rec["class_name"],
            "reference": ref,
            "predicted": pred,
            "rouge_l":   round(r_score, 4),
        })

        if (i + 1) % 25 == 0:
            print(f"  Evaluated {i+1}/{len(val_data)}...")

    # ── Aggregate scores ──────────────────────────────────────────────────────
    smooth  = SmoothingFunction().method1
    bleu1   = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0),            smoothing_function=smooth)
    bleu4   = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    rouge_l = sum(rouge_scores) / len(rouge_scores)

    print(f"\n{'='*50}")
    print(f"  BLEU-1   : {bleu1:.4f}   (target > 0.40)")
    print(f"  BLEU-4   : {bleu4:.4f}   (target > 0.25)")
    print(f"  ROUGE-L  : {rouge_l:.4f}   (target > 0.40)")
    print(f"{'='*50}")

    if bleu1 >= 0.40 and bleu4 >= 0.25 and rouge_l >= 0.40:
        print("\n✓ All targets met! Model is performing well.")
    else:
        print("\nSome targets not met. Consider:")
        if bleu1 < 0.40:
            print("  - BLEU-1 low → model missing key vocabulary. Try more epochs.")
        if bleu4 < 0.25:
            print("  - BLEU-4 low → phrase-level mismatch. Increase LoRA rank or LR.")
        if rouge_l < 0.40:
            print("  - ROUGE-L low → sequence overlap weak. Check generation params.")

    # ── Save detailed results ─────────────────────────────────────────────────
    output = {
        "summary": {
            "n_evaluated": len(val_data),
            "bleu_1":  round(bleu1,   4),
            "bleu_4":  round(bleu4,   4),
            "rouge_l": round(rouge_l, 4),
        },
        "samples": per_sample,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
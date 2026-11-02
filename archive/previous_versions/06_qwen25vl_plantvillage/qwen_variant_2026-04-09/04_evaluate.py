# 04_evaluate.py
# Comprehensive evaluation for fine-tuned Qwen2.5-VL-7B-Instruct
# Metrics: BLEU-1, BLEU-4, ROUGE-L, Condition Accuracy, Macro F1,
#          Precision, Recall, Severity R², CLIP Similarity, Per-class F1

import json
import re
import torch
import nltk
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix
)
from scipy.stats import pearsonr
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    CLIPProcessor,
    CLIPModel,
)
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_module

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Config ────────────────────────────────────────────────────────────────
ADAPTER_PATH    = "outputs/qwen25vl_plantvillage/best"
CAPTIONS        = "captions.json"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_EVAL        = 200    # set to None to evaluate on entire val set
OUTPUT_REPORT   = "evaluation_report.json"

EVAL_INSTRUCTION = (
    "Analyze this plant leaf image and provide a detailed diagnosis. "
    "Include: plant name, disease or health condition, severity level, "
    "causal pathogen, and visible symptoms."
)

# Canonical condition labels for classification metrics
CONDITION_LABELS = [
    "Healthy",
    "Late Blight",
    "Early Blight",
    "Bacterial Spot",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mite Damage",
    "Target Spot",
    "Mosaic Virus",
    "Yellow Leaf Curl Virus",
]

SEVERITY_SCALE = {
    "none":               0.0,
    "mild":               1.0,
    "mild to moderate":   1.5,
    "moderate":           2.0,
    "moderate to severe": 2.5,
    "severe":             3.0,
}


# ── Extraction helpers ────────────────────────────────────────────────────

def extract_condition(text: str) -> str:
    """Extract the condition label from a structured caption or generated text."""
    text_l = text.lower()
    # Try structured format first: "Condition: X."
    m = re.search(r"condition[:\s]+([^.]+)", text_l)
    if m:
        cand = m.group(1).strip().title()
        # Match against known labels
        for label in CONDITION_LABELS:
            if label.lower() in cand.lower() or cand.lower() in label.lower():
                return label
    # Fallback: keyword scan
    for label in CONDITION_LABELS:
        if label.lower() in text_l:
            return label
    return "Unknown"


def extract_severity(text: str) -> float:
    """Extract numeric severity from text, returns 0.0 if not found."""
    text_l = text.lower()
    for key, val in sorted(SEVERITY_SCALE.items(), key=lambda x: -len(x[0])):
        if key in text_l:
            return val
    return 0.0


# ── Model loaders ─────────────────────────────────────────────────────────

def load_model():
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
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print("Model ready.\n")
    return model, processor


def load_clip():
    print("Loading CLIP for similarity scoring...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to("cuda")
    clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()
    return clip_model, clip_proc


# ── Inference helper ──────────────────────────────────────────────────────

def generate_caption(image: Image.Image, model, processor) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": EVAL_INSTRUCTION},
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
            max_new_tokens=160,
            num_beams=4,
            repetition_penalty=2.5,
            no_repeat_ngram_size=6,
            early_stopping=True,
        )
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def clip_similarity(image: Image.Image, text: str, clip_model, clip_proc) -> float:
    inputs = clip_proc(
        text=[text], images=image, return_tensors="pt", padding=True
    ).to("cuda")
    with torch.no_grad():
        out     = clip_model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return (img_emb * txt_emb).sum().item()


# ── Main evaluation ───────────────────────────────────────────────────────

def main():
    with open(CAPTIONS) as f:
        val_data = json.load(f)["val"]
    if MAX_EVAL:
        val_data = val_data[:MAX_EVAL]
    print(f"Evaluating on {len(val_data)} validation samples\n")

    model,      processor  = load_model()
    clip_model, clip_proc  = load_clip()

    rouge_scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)

    refs           = []     # for BLEU
    hyps           = []     # for BLEU
    rouge_scores   = []
    clip_scores    = []
    pred_conditions = []
    true_conditions = []
    pred_severities = []
    true_severities = []
    per_class_preds = defaultdict(list)   # class_name -> list of (true, pred)

    for i, rec in enumerate(val_data):
        image    = Image.open(rec["image"]).convert("RGB")
        ref_cap  = rec["caption"]
        class_nm = rec.get("class_name", "")

        # Generate
        pred_cap = generate_caption(image, model, processor)

        # BLEU/ROUGE
        refs.append([ref_cap.lower().split()])
        hyps.append(pred_cap.lower().split())
        rouge_scores.append(
            rouge_scorer.score(ref_cap, pred_cap)["rougeL"].fmeasure
        )

        # CLIP
        clip_scores.append(clip_similarity(image, pred_cap, clip_model, clip_proc))

        # Condition classification
        true_cond = extract_condition(ref_cap)
        pred_cond = extract_condition(pred_cap)
        true_conditions.append(true_cond)
        pred_conditions.append(pred_cond)

        # Severity regression
        true_sev = extract_severity(ref_cap)
        pred_sev = extract_severity(pred_cap)
        true_severities.append(true_sev)
        pred_severities.append(pred_sev)

        # Per-class tracking
        if class_nm:
            per_class_preds[class_nm].append((true_cond, pred_cond))

        if (i + 1) % 25 == 0:
            print(f"  Evaluated {i + 1} / {len(val_data)}...")

    # ── Compute metrics ───────────────────────────────────────────────────
    smooth = SmoothingFunction().method1
    bleu1  = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0),           smoothing_function=smooth)
    bleu4  = corpus_bleu(refs, hyps, weights=(.25, .25, .25, .25),   smoothing_function=smooth)
    rougeL = float(np.mean(rouge_scores))
    clip_s = float(np.mean(clip_scores))

    # Condition accuracy
    accuracy = sum(p == t for p, t in zip(pred_conditions, true_conditions)) / len(true_conditions)

    # Filter to known labels for sklearn metrics
    known_mask  = [t in CONDITION_LABELS for t in true_conditions]
    t_known     = [t for t, k in zip(true_conditions, known_mask) if k]
    p_known     = [p for p, k in zip(pred_conditions, known_mask) if k]
    labels_used = sorted(set(t_known))

    macro_f1  = f1_score(t_known,  p_known, labels=labels_used, average="macro",  zero_division=0)
    macro_pre = precision_score(t_known, p_known, labels=labels_used, average="macro",  zero_division=0)
    macro_rec = recall_score(t_known,   p_known, labels=labels_used, average="macro",  zero_division=0)

    # Per-class F1
    per_class_f1 = {}
    for label in labels_used:
        y_true = [1 if t == label else 0 for t in t_known]
        y_pred = [1 if p == label else 0 for p in p_known]
        per_class_f1[label] = round(f1_score(y_true, y_pred, zero_division=0), 4)

    # Severity R²
    sev_r2 = float("nan")
    if len(set(true_severities)) > 1:
        try:
            corr, _ = pearsonr(true_severities, pred_severities)
            sev_r2  = round(corr ** 2, 4)
        except Exception:
            pass
    sev_mae = float(np.mean(np.abs(np.array(true_severities) - np.array(pred_severities))))

    # ── Print report ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  EVALUATION REPORT — Qwen2.5-VL-7B PlantVillage")
    print(f"{'='*55}")
    print(f"\n  Caption Quality")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  BLEU-1    : {bleu1:.4f}   (target > 0.40)")
    print(f"  BLEU-4    : {bleu4:.4f}   (target > 0.25)")
    print(f"  ROUGE-L   : {rougeL:.4f}   (target > 0.40)")
    print(f"  CLIP sim  : {clip_s:.4f}   (target > 0.25)")
    print(f"\n  Disease Classification")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Accuracy  : {accuracy:.4f}   (target > 0.85)")
    print(f"  Macro F1  : {macro_f1:.4f}   (target > 0.80)")
    print(f"  Precision : {macro_pre:.4f}")
    print(f"  Recall    : {macro_rec:.4f}")
    print(f"\n  Severity Estimation")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Severity R²  : {sev_r2:.4f}   (target > 0.70)")
    print(f"  Severity MAE : {sev_mae:.4f}   (scale 0-3)")
    print(f"\n  Per-class F1")
    print(f"  ─────────────────────────────────────────────────")
    for cls, f1 in sorted(per_class_f1.items(), key=lambda x: -x[1]):
        bar = "█" * int(f1 * 20)
        print(f"  {cls:<28} {f1:.4f}  {bar}")
    print(f"{'='*55}\n")

    # ── Save report ───────────────────────────────────────────────────────
    report = {
        "model": "Qwen2.5-VL-7B-Instruct (QLoRA r64)",
        "n_eval": len(val_data),
        "caption_quality": {
            "bleu1": round(bleu1, 4),
            "bleu4": round(bleu4, 4),
            "rougeL": round(rougeL, 4),
            "clip_similarity": round(clip_s, 4),
        },
        "classification": {
            "accuracy":  round(accuracy,  4),
            "macro_f1":  round(macro_f1,  4),
            "precision": round(macro_pre, 4),
            "recall":    round(macro_rec, 4),
        },
        "severity": {
            "r2":  sev_r2,
            "mae": round(sev_mae, 4),
        },
        "per_class_f1": per_class_f1,
    }
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()

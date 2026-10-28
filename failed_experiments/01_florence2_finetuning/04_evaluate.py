# 04_evaluate.py  (v2 — full evaluation suite)
#
# METRICS:
#   Caption quality  : BLEU-1, BLEU-4, ROUGE-L (text generation quality)
#   Classification   : Accuracy, Macro F1, Precision, Recall (per-class and overall)
#   Regression proxy : R² on severity ordinal encoding (how well severity ordering is preserved)
#   Confidence       : CLIP cosine similarity distribution across val set
#
# USAGE:
#   python 04_evaluate.py --model blip2
#   python 04_evaluate.py --model florence2
#   python 04_evaluate.py --model blip2 --compare florence2   # side-by-side table
#
# NOTE: 04_evaluate.py deliberately re-uses the same CLIP OOD detector from 03_inference.py
#       so the similarity scores are consistent across the two scripts.

import argparse
import json
import re
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    r2_score,
    confusion_matrix,
)
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_module
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    CLIPProcessor,
    CLIPModel,
)
from peft import PeftModel

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BLIP2_BASE        = "Salesforce/blip2-opt-2.7b"
BLIP2_ADAPTER     = "outputs/blip2_plantvillage/best"
FLORENCE2_BASE    = "microsoft/Florence-2-base-ft"
FLORENCE2_ADAPTER = "outputs/florence2_plantvillage/best"
CLIP_MODEL_ID     = "openai/clip-vit-base-patch32"
CAPTIONS_FILE     = "captions.json"
MAX_EVAL          = 300    # set to None to evaluate full val set

# ── Severity → ordinal encoding for R² ───────────────────────────────────────
SEVERITY_ORDINAL = {
    "none":                    0,
    "mild":                    1,
    "mild to moderate":        2,
    "moderate":                3,
    "moderate to severe":      4,
    "severe":                  5,
}

# ── Caption parser (mirrors 03_inference.py) ──────────────────────────────────
STRUCTURED_PATTERN = re.compile(
    r"Plant:\s*(?P<plant>[^.]+)\.\s*"
    r"Condition:\s*(?P<condition>[^.]+)\.\s*"
    r"Severity:\s*(?P<severity>[^.]+)\.",
    re.IGNORECASE,
)

def parse_caption(caption: str):
    m = STRUCTURED_PATTERN.search(caption)
    if not m:
        return None
    return (
        m.group("plant").strip().lower(),
        m.group("condition").strip().lower(),
        m.group("severity").strip().lower(),
    )


# ── Model loaders ──────────────────────────────────────────────────────────────
def load_blip2():
    processor = Blip2Processor.from_pretrained(BLIP2_ADAPTER)
    base      = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_BASE, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, BLIP2_ADAPTER)
    model.eval()
    return model, processor, "BLIP-2 OPT-2.7B"


def load_florence2():
    processor = AutoProcessor.from_pretrained(FLORENCE2_ADAPTER, trust_remote_code=True)
    base      = AutoModelForCausalLM.from_pretrained(
        FLORENCE2_BASE, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, FLORENCE2_ADAPTER)
    model.eval()
    return model, processor, "Florence-2-base"


def generate_caption_blip2(image, model, processor):
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
    raw   = processor.decode(out[0], skip_special_tokens=True).strip()
    sents = raw.split(". ")
    clean = ". ".join(sents[:3]).strip()
    return clean if clean.endswith(".") else clean + "."


def generate_caption_florence2(image, model, processor):
    inputs = processor(text="<CAPTION>", images=image, return_tensors="pt").to("cuda", torch.bfloat16)
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
    raw   = processor.decode(out[0], skip_special_tokens=True).strip().replace("<CAPTION>", "").strip()
    sents = raw.split(". ")
    clean = ". ".join(sents[:3]).strip()
    return clean if clean.endswith(".") else clean + "."


# ── CLIP similarity ────────────────────────────────────────────────────────────
def load_clip():
    proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to("cuda").half()
    model.eval()
    return proc, model


@torch.no_grad()
def clip_similarity(image, text, clip_proc, clip_model):
    inputs = clip_proc(text=[text], images=image, return_tensors="pt", padding=True).to("cuda")
    out    = clip_model(**inputs)
    return float(out.logits_per_image.squeeze().cpu())


# ── Evaluation core ────────────────────────────────────────────────────────────
def evaluate_model(model_key: str, val_data: list):
    if model_key == "blip2":
        model, processor, model_name = load_blip2()
        gen_fn = lambda img: generate_caption_blip2(img, model, processor)
    else:
        model, processor, model_name = load_florence2()
        gen_fn = lambda img: generate_caption_florence2(img, model, processor)

    clip_proc, clip_model = load_clip()

    rouge_scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    smooth       = SmoothingFunction().method1

    refs         = []
    hyps         = []
    rouge_scores = []

    gt_conditions  = []
    pred_conditions = []
    gt_severities  = []
    pred_severities = []
    clip_sims       = []

    parse_failures = 0

    print(f"\n{'='*65}")
    print(f"  Evaluating: {model_name}  ({len(val_data)} samples)")
    print(f"{'='*65}")

    for i, rec in enumerate(val_data):
        image  = Image.open(rec["image"]).convert("RGB")
        ref    = rec["caption"]
        pred   = gen_fn(image)

        # BLEU / ROUGE tokens
        refs.append([ref.lower().split()])
        hyps.append(pred.lower().split())
        rouge_scores.append(rouge_scorer.score(ref, pred)["rougeL"].fmeasure)

        # CLIP sim: compare pred caption with image
        sim = clip_similarity(image, pred, clip_proc, clip_model)
        clip_sims.append(sim)

        # Parse structured fields
        gt_parsed   = parse_caption(ref)
        pred_parsed = parse_caption(pred)

        if gt_parsed and pred_parsed:
            _, gt_cond,   gt_sev   = gt_parsed
            _, pred_cond, pred_sev = pred_parsed
            gt_conditions.append(gt_cond)
            pred_conditions.append(pred_cond)
            gt_severities.append(SEVERITY_ORDINAL.get(gt_sev,   3))
            pred_severities.append(SEVERITY_ORDINAL.get(pred_sev, 3))
        else:
            parse_failures += 1

        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i+1}/{len(val_data)}...")

    # ── Text metrics ────────────────────────────────────────────────────────────
    bleu1   = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4   = corpus_bleu(refs, hyps, weights=(.25, .25, .25, .25), smoothing_function=smooth)
    rouge_l = float(np.mean(rouge_scores))

    # ── Classification metrics ──────────────────────────────────────────────────
    if gt_conditions:
        acc  = accuracy_score(gt_conditions, pred_conditions)
        f1   = f1_score(gt_conditions, pred_conditions, average="macro", zero_division=0)
        prec = precision_score(gt_conditions, pred_conditions, average="macro", zero_division=0)
        rec  = recall_score(gt_conditions, pred_conditions, average="macro", zero_division=0)
        per_class_report = classification_report(
            gt_conditions, pred_conditions,
            zero_division=0, output_dict=True
        )
    else:
        acc = f1 = prec = rec = 0.0
        per_class_report = {}

    # ── Severity R² ─────────────────────────────────────────────────────────────
    r2 = r2_score(gt_severities, pred_severities) if len(gt_severities) > 1 else 0.0

    # ── CLIP similarity stats ───────────────────────────────────────────────────
    clip_mean = float(np.mean(clip_sims))
    clip_std  = float(np.std(clip_sims))

    results = {
        "model":          model_name,
        "n_evaluated":    len(val_data),
        "parse_failures": parse_failures,

        # Text quality
        "bleu1":          round(bleu1,   4),
        "bleu4":          round(bleu4,   4),
        "rouge_l":        round(rouge_l, 4),

        # Classification
        "accuracy":       round(acc,  4),
        "macro_f1":       round(f1,   4),
        "macro_precision":round(prec, 4),
        "macro_recall":   round(rec,  4),

        # Regression
        "severity_r2":    round(r2,   4),

        # CLIP
        "clip_sim_mean":  round(clip_mean, 4),
        "clip_sim_std":   round(clip_std,  4),

        # Per-class
        "per_class_f1":   {
            k: round(v["f1-score"], 4)
            for k, v in per_class_report.items()
            if isinstance(v, dict) and "f1-score" in v
        },
    }

    return results


# ── Pretty printer ──────────────────────────────────────────────────────────────
def print_results(res: dict):
    name = res["model"]
    print(f"\n{'='*65}")
    print(f"  Results — {name}")
    print(f"{'='*65}")
    print(f"  Samples evaluated : {res['n_evaluated']}")
    print(f"  Parse failures    : {res['parse_failures']}")
    print()
    print(f"  ── Caption Quality ──────────────────────────────")
    print(f"  BLEU-1            : {res['bleu1']:.4f}  (target > 0.40)")
    print(f"  BLEU-4            : {res['bleu4']:.4f}  (target > 0.25)")
    print(f"  ROUGE-L           : {res['rouge_l']:.4f}  (target > 0.40)")
    print()
    print(f"  ── Classification ───────────────────────────────")
    print(f"  Accuracy          : {res['accuracy']:.4f}  (target > 0.80)")
    print(f"  Macro F1          : {res['macro_f1']:.4f}  (target > 0.75)")
    print(f"  Macro Precision   : {res['macro_precision']:.4f}")
    print(f"  Macro Recall      : {res['macro_recall']:.4f}")
    print()
    print(f"  ── Severity (Regression) ────────────────────────")
    print(f"  Severity R²       : {res['severity_r2']:.4f}  (target > 0.70)")
    print()
    print(f"  ── CLIP Image-Caption Alignment ─────────────────")
    print(f"  CLIP sim mean     : {res['clip_sim_mean']:.4f}")
    print(f"  CLIP sim std      : {res['clip_sim_std']:.4f}")
    print()
    print(f"  ── Per-class F1 ─────────────────────────────────")
    for cls, score in sorted(res["per_class_f1"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 20)
        print(f"  {cls:<35} {score:.4f}  {bar}")
    print(f"{'='*65}")


def compare_models(res_a: dict, res_b: dict):
    keys = ["bleu1", "bleu4", "rouge_l", "accuracy", "macro_f1",
            "macro_precision", "macro_recall", "severity_r2",
            "clip_sim_mean"]
    labels = {
        "bleu1":          "BLEU-1",
        "bleu4":          "BLEU-4",
        "rouge_l":        "ROUGE-L",
        "accuracy":       "Accuracy",
        "macro_f1":       "Macro F1",
        "macro_precision":"Macro Precision",
        "macro_recall":   "Macro Recall",
        "severity_r2":    "Severity R²",
        "clip_sim_mean":  "CLIP Sim Mean",
    }
    col_a = res_a["model"][:20]
    col_b = res_b["model"][:20]

    print(f"\n{'='*65}")
    print(f"  Model Comparison")
    print(f"{'='*65}")
    print(f"  {'Metric':<22} {col_a:<22} {col_b:<20} Winner")
    print(f"  {'-'*60}")
    for k in keys:
        va, vb = res_a[k], res_b[k]
        winner = col_a if va >= vb else col_b
        delta  = abs(va - vb)
        print(f"  {labels[k]:<22} {va:<22.4f} {vb:<20.4f} {winner} (+{delta:.4f})")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="PlantVillage VLM Evaluation v2")
    parser.add_argument("--model",   choices=["blip2", "florence2"], default="blip2")
    parser.add_argument("--compare", choices=["blip2", "florence2"], default=None,
                        help="Second model to compare against")
    parser.add_argument("--max_eval", type=int, default=MAX_EVAL,
                        help="Max validation samples to evaluate (default 300)")
    args = parser.parse_args()

    with open(CAPTIONS_FILE) as f:
        val_data = json.load(f)["val"]
    if args.max_eval:
        val_data = val_data[:args.max_eval]

    print(f"Loaded {len(val_data)} validation samples.")

    res_a = evaluate_model(args.model, val_data)
    print_results(res_a)

    all_results = [res_a]

    if args.compare:
        res_b = evaluate_model(args.compare, val_data)
        print_results(res_b)
        compare_models(res_a, res_b)
        all_results.append(res_b)

    out_file = "evaluation_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results if args.compare else res_a, f, indent=2)
    print(f"\nResults saved → {out_file}")


if __name__ == "__main__":
    main()
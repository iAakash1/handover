# quick_eval_florence2.py
# Place in: C:\Users\Admin\Desktop\VLM\florence\
# Run AFTER quick_train_florence2.py completes.
#
# Run:
#   python quick_eval_florence2.py

import json, torch, re
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_module
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, r2_score, mean_absolute_error
)

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE_MODEL   = "microsoft/Florence-2-base-ft"
ADAPTER_PATH = "outputs/florence2_plantvillage/best"
CAPTIONS     = "../tp/captions.json"
EVAL_OUTPUT  = "evaluation_report_florence2.json"
MAX_SAMPLES  = 200
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL   = "openai/clip-vit-base-patch32"
TASK_TOKEN   = "<CAPTION>"

SEVERITY_SCALE = {
    "none": 0.0, "mild": 1.0, "mild to moderate": 2.0,
    "moderate": 3.0, "moderate to severe": 4.0, "severe": 5.0,
}


def extract_field(text, field):
    m = re.search(rf"{re.escape(field)}:\s*(.+?)(?:\.|$|\n)", text, re.IGNORECASE)
    return m.group(1).strip() if m else "Unknown"


def parse_condition(t): return extract_field(t, "Condition")
def parse_severity(t):  return extract_field(t, "Severity")


def severity_to_score(text):
    low = text.lower().strip()
    for k, v in SEVERITY_SCALE.items():
        if k in low:
            return v
    return 3.0


def main():
    if not Path(ADAPTER_PATH).exists():
        print("ERROR: No model at", ADAPTER_PATH)
        print("Run quick_train_florence2.py first.")
        return

    if not Path(CAPTIONS).exists():
        print("ERROR: captions.json not found at", CAPTIONS)
        return

    with open(CAPTIONS) as f:
        data = json.load(f)
    val_records = data.get("val", data)[:MAX_SAMPLES]
    print("Evaluating on", len(val_records), "val samples...")

    print("Loading Florence-2 + adapter...")
    processor = AutoProcessor.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print("Model loaded. Running inference...\n")

    refs, hyps = [], []
    for i, rec in enumerate(val_records):
        image  = Image.open(rec["image"]).convert("RGB")
        inputs = processor(
            text=TASK_TOKEN, images=image, return_tensors="pt"
        ).to(DEVICE, torch.bfloat16)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=160,
                do_sample=False,
                num_beams=4,
            )

        pred = processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()

        refs.append(rec["caption"])
        hyps.append(pred)

        if (i + 1) % 25 == 0:
            print("  [{}/{}] done".format(i + 1, len(val_records)))
            print("  Sample:", pred[:80])

    print("\nComputing metrics...")
    smooth   = SmoothingFunction().method1
    refs_tok = [r.lower().split() for r in refs]
    hyps_tok = [h.lower().split() for h in hyps]
    bleu1 = corpus_bleu([[r] for r in refs_tok], hyps_tok,
                         weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu([[r] for r in refs_tok], hyps_tok,
                         weights=(.25, .25, .25, .25), smoothing_function=smooth)
    scorer  = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = float(np.mean([scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]))

    true_conds = [parse_condition(r) for r in refs]
    pred_conds = [parse_condition(h) for h in hyps]
    true_sevs  = [parse_severity(r)  for r in refs]
    pred_sevs  = [parse_severity(h)  for h in hyps]

    labels    = sorted(set(true_conds + pred_conds))
    accuracy  = float(accuracy_score(true_conds, pred_conds))
    macro_f1  = float(f1_score(true_conds, pred_conds, average="macro", zero_division=0, labels=labels))
    precision = float(precision_score(true_conds, pred_conds, average="macro", zero_division=0, labels=labels))
    recall    = float(recall_score(true_conds, pred_conds, average="macro", zero_division=0, labels=labels))

    y_true  = [severity_to_score(s) for s in true_sevs]
    y_pred  = [severity_to_score(s) for s in pred_sevs]
    sev_r2  = float(r2_score(y_true, y_pred))
    sev_mae = float(mean_absolute_error(y_true, y_pred))

    print("Loading CLIP for text similarity...")
    from transformers import CLIPModel, CLIPProcessor as CP
    clip_m = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True).to(DEVICE)
    clip_p = CP.from_pretrained(CLIP_MODEL)
    clip_m.eval()
    clip_scores = []
    for ref, hyp in zip(refs, hyps):
        inp = clip_p(
            text=[ref, hyp], return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        ).to(DEVICE)
        with torch.no_grad():
            out   = clip_m.text_model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
            )
            feats = out.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        clip_scores.append(float((feats[0] * feats[1]).sum().item()))

    report = {
        "model":      "Florence-2-base-ft (LoRA r16, 3 epochs)",
        "n_samples":  len(val_records),
        "caption":    {"bleu1": round(bleu1, 4), "bleu4": round(bleu4, 4), "rougeL": round(rouge_l, 4)},
        "classification": {
            "accuracy":  round(accuracy, 4),
            "macro_f1":  round(macro_f1, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "per_class_f1": {
                cls: round(float(f1_score(true_conds, pred_conds, labels=[cls],
                                          average="macro", zero_division=0)), 4)
                for cls in labels
            },
        },
        "severity":   {"r2": round(sev_r2, 4), "mae": round(sev_mae, 4)},
        "clip":       {
            "mean":   round(float(np.mean(clip_scores)), 4),
            "median": round(float(np.median(clip_scores)), 4),
            "std":    round(float(np.std(clip_scores)), 4),
        },
    }

    with open(EVAL_OUTPUT, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 50)
    print("  Florence-2 Evaluation  ({} samples)".format(len(val_records)))
    print("=" * 50)
    print("  BLEU-1   :", report["caption"]["bleu1"])
    print("  BLEU-4   :", report["caption"]["bleu4"])
    print("  ROUGE-L  :", report["caption"]["rougeL"])
    print("  Accuracy :", report["classification"]["accuracy"])
    print("  Macro F1 :", report["classification"]["macro_f1"])
    print("  Precision:", report["classification"]["precision"])
    print("  Recall   :", report["classification"]["recall"])
    print("  Sev R2   :", report["severity"]["r2"])
    print("  Sev MAE  :", report["severity"]["mae"])
    print("=" * 50)
    print("Saved ->", EVAL_OUTPUT)


if __name__ == "__main__":
    main()

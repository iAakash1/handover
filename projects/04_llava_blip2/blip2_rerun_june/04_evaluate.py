# 04_evaluate.py (BLIP-2 / tp folder)
import json
import torch
import nltk
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_module
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import re

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

INFERENCE_FILE  = "inference_results.json"
EVAL_OUTPUT     = "evaluation_report.json"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

SEVERITY_SCALE = {
    "none": 0.0, "mild": 1.0, "mild to moderate": 2.0,
    "moderate": 3.0, "moderate to severe": 4.0, "severe": 5.0,
}

def extract_field(text, field):
    m = re.search(rf"{re.escape(field)}:\s*(.+?)(?:\.|$|\n)", text, re.IGNORECASE)
    return m.group(1).strip() if m else "Unknown"

def parse_condition(text): return extract_field(text, "Condition")
def parse_severity(text):  return extract_field(text, "Severity")

def severity_to_score(text):
    low = text.lower().strip()
    for k, v in SEVERITY_SCALE.items():
        if k in low:
            return v
    return 3.0

def compute_clip_scores(refs, hyps, clip_model, clip_processor):
    scores = []
    for ref, hyp in zip(refs, hyps):
        inputs = clip_processor(
            text=[ref, hyp], return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            feats = outputs.pooler_output
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        cosine = float((feats[0] * feats[1]).sum().item())
        scores.append(cosine)
    return scores

def main():
    if not Path(INFERENCE_FILE).exists():
        print(f"ERROR: {INFERENCE_FILE} not found. Run 03_inference.py first.")
        return

    with open(INFERENCE_FILE) as f:
        raw = json.load(f)

    records = raw if isinstance(raw, list) else raw.get("val_set", [])
    print(f"Evaluating {len(records)} samples...\n")

    refs = [r.get("reference", r.get("caption", "")) for r in records]
    hyps = [r.get("prediction", r.get("caption", "")) for r in records]
    true_conds = [parse_condition(r) for r in refs]
    pred_conds = [parse_condition(r) for r in hyps]
    true_sevs  = [parse_severity(r)  for r in refs]
    pred_sevs  = [parse_severity(r)  for r in hyps]

    smooth  = SmoothingFunction().method1
    refs_tok = [r.lower().split() for r in refs]
    hyps_tok = [h.lower().split() for h in hyps]
    bleu1 = corpus_bleu([[r] for r in refs_tok], hyps_tok, weights=(1,0,0,0),         smoothing_function=smooth)
    bleu4 = corpus_bleu([[r] for r in refs_tok], hyps_tok, weights=(.25,.25,.25,.25),  smoothing_function=smooth)

    scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = float(np.mean([scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]))

    labels = sorted(set(true_conds + pred_conds))
    accuracy  = float(accuracy_score(true_conds, pred_conds))
    macro_f1  = float(f1_score(true_conds, pred_conds, average="macro", zero_division=0, labels=labels))
    precision = float(precision_score(true_conds, pred_conds, average="macro", zero_division=0, labels=labels))
    recall    = float(recall_score(true_conds, pred_conds, average="macro", zero_division=0, labels=labels))

    y_true_num = [severity_to_score(s) for s in true_sevs]
    y_pred_num = [severity_to_score(s) for s in pred_sevs]
    sev_r2  = float(r2_score(y_true_num, y_pred_num))
    sev_mae = float(mean_absolute_error(y_true_num, y_pred_num))

    print("[CLIP] Loading CLIP for text similarity...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to(DEVICE)
    clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()
    clip_scores = compute_clip_scores(refs, hyps, clip_model, clip_proc)

    report = {
        "model": "BLIP-2 OPT-2.7B",
        "n_samples": len(records),
        "caption": {"bleu1": round(bleu1,4), "bleu4": round(bleu4,4), "rougeL": round(rouge_l,4)},
        "classification": {
            "accuracy": round(accuracy,4), "macro_f1": round(macro_f1,4),
            "precision": round(precision,4), "recall": round(recall,4),
            "per_class_f1": {
                cls: round(float(f1_score(true_conds, pred_conds, labels=[cls], average="macro", zero_division=0)), 4)
                for cls in labels
            }
        },
        "severity": {"r2": round(sev_r2,4), "mae": round(sev_mae,4)},
        "clip": {
            "mean":   round(float(np.mean(clip_scores)),4),
            "median": round(float(np.median(clip_scores)),4),
            "std":    round(float(np.std(clip_scores)),4),
        }
    }

    with open(EVAL_OUTPUT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  BLIP-2 OPT-2.7B Evaluation  ({len(records)} samples)")
    print(f"{'='*50}")
    print(f"  BLEU-1   : {report['caption']['bleu1']}")
    print(f"  BLEU-4   : {report['caption']['bleu4']}")
    print(f"  ROUGE-L  : {report['caption']['rougeL']}")
    print(f"  Accuracy : {report['classification']['accuracy']}")
    print(f"  Macro F1 : {report['classification']['macro_f1']}")
    print(f"  Sev R2   : {report['severity']['r2']}")
    print(f"  Sev MAE  : {report['severity']['mae']}")
    print(f"{'='*50}")
    print(f"Saved to {EVAL_OUTPUT}")

if __name__ == "__main__":
    main()
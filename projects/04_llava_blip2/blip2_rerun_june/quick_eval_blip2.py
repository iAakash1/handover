# quick_eval_blip2.py
# Run AFTER quick_train_blip2.py completes.
# Loads the saved BLIP-2 adapter, runs inference on val set,
# computes all metrics, saves evaluation_report.json

import json, torch, re
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
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

BASE_MODEL   = "Salesforce/blip2-opt-2.7b"
ADAPTER_PATH = "outputs/blip2_plantvillage/best"
CAPTIONS     = "captions.json"
EVAL_OUTPUT  = "evaluation_report.json"
MAX_SAMPLES  = 200
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL   = "openai/clip-vit-base-patch32"

SEVERITY_SCALE = {
    "none":0.0,"mild":1.0,"mild to moderate":2.0,
    "moderate":3.0,"moderate to severe":4.0,"severe":5.0,
}

PROMPT = (
    "Question: Describe this plant leaf. Include plant name, "
    "disease condition, severity, and symptoms. Answer:"
)

def extract_field(text, field):
    m = re.search(rf"{re.escape(field)}:\s*(.+?)(?:\.|$|\n)", text, re.IGNORECASE)
    return m.group(1).strip() if m else "Unknown"

def parse_condition(t): return extract_field(t, "Condition")
def parse_severity(t):  return extract_field(t, "Severity")

def severity_to_score(text):
    low = text.lower().strip()
    for k, v in SEVERITY_SCALE.items():
        if k in low: return v
    return 3.0

def main():
    if not Path(ADAPTER_PATH).exists():
        print(f"ERROR: No model at {ADAPTER_PATH}")
        print("Run quick_train_blip2.py first.")
        return

    if not Path(CAPTIONS).exists():
        print("ERROR: captions.json not found.")
        return

    with open(CAPTIONS) as f:
        data = json.load(f)
    val_records = data.get("val", data)[:MAX_SAMPLES]
    print(f"Evaluating on {len(val_records)} val samples...")

    print("Loading BLIP-2 + adapter...")
    processor = Blip2Processor.from_pretrained(ADAPTER_PATH)
    base = Blip2ForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print("Model loaded. Running inference...\n")

    refs, hyps = [], []
    for i, rec in enumerate(val_records):
        image = Image.open(rec["image"]).convert("RGB")
        inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(DEVICE, torch.bfloat16)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=80, num_beams=4,
                repetition_penalty=3.0, no_repeat_ngram_size=8,
                length_penalty=0.6, early_stopping=True,
            )
        pred = processor.decode(out[0], skip_special_tokens=True).strip()
        refs.append(rec["caption"])
        hyps.append(pred)
        if (i+1) % 25 == 0:
            print(f"  [{i+1}/{len(val_records)}] done")

    print("\nComputing metrics...")
    smooth   = SmoothingFunction().method1
    refs_tok = [r.lower().split() for r in refs]
    hyps_tok = [h.lower().split() for h in hyps]
    bleu1 = corpus_bleu([[r] for r in refs_tok], hyps_tok,
                         weights=(1,0,0,0), smoothing_function=smooth)
    bleu4 = corpus_bleu([[r] for r in refs_tok], hyps_tok,
                         weights=(.25,.25,.25,.25), smoothing_function=smooth)
    scorer  = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = float(np.mean([scorer.score(r,h)["rougeL"].fmeasure for r,h in zip(refs,hyps)]))

    true_conds = [parse_condition(r) for r in refs]
    pred_conds = [parse_condition(h) for h in hyps]
    true_sevs  = [parse_severity(r)  for r in refs]
    pred_sevs  = [parse_severity(h)  for h in hyps]

    labels    = sorted(set(true_conds + pred_conds))
    accuracy  = float(accuracy_score(true_conds, pred_conds))
    macro_f1  = float(f1_score(true_conds, pred_conds, average="macro", zero_division=0, labels=labels))
    precision = float(precision_score(true_conds, pred_conds, average="macro", zero_division=0, labels=labels))
    recall    = float(recall_score(true_conds, pred_conds, average="macro", zero_division=0, labels=labels))

    y_true = [severity_to_score(s) for s in true_sevs]
    y_pred = [severity_to_score(s) for s in pred_sevs]
    sev_r2  = float(r2_score(y_true, y_pred))
    sev_mae = float(mean_absolute_error(y_true, y_pred))

    print("Loading CLIP for text similarity...")
    from transformers import CLIPModel, CLIPProcessor as CP
    clip_m = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True).to(DEVICE)
    clip_p = CP.from_pretrained(CLIP_MODEL)
    clip_m.eval()
    clip_scores = []
    for ref, hyp in zip(refs, hyps):
        inp = clip_p(text=[ref,hyp], return_tensors="pt",
                     padding=True, truncation=True, max_length=77).to(DEVICE)
        with torch.no_grad():
            out = clip_m.text_model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
            )
            feats = out.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        clip_scores.append(float((feats[0]*feats[1]).sum().item()))

    report = {
        "model": "BLIP-2 OPT-2.7B (1 epoch LoRA r32)",
        "n_samples": len(val_records),
        "caption":  {"bleu1": round(bleu1,4), "bleu4": round(bleu4,4), "rougeL": round(rouge_l,4)},
        "classification": {
            "accuracy":  round(accuracy,4),
            "macro_f1":  round(macro_f1,4),
            "precision": round(precision,4),
            "recall":    round(recall,4),
            "per_class_f1": {
                cls: round(float(f1_score(true_conds, pred_conds, labels=[cls],
                                          average="macro", zero_division=0)),4)
                for cls in labels
            },
        },
        "severity": {"r2": round(sev_r2,4), "mae": round(sev_mae,4)},
        "clip": {
            "mean":   round(float(np.mean(clip_scores)),4),
            "median": round(float(np.median(clip_scores)),4),
            "std":    round(float(np.std(clip_scores)),4),
        },
    }

    with open(EVAL_OUTPUT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  BLIP-2 OPT-2.7B Evaluation  ({len(val_records)} samples)")
    print(f"{'='*50}")
    print(f"  BLEU-1   : {report['caption']['bleu1']}")
    print(f"  BLEU-4   : {report['caption']['bleu4']}")
    print(f"  ROUGE-L  : {report['caption']['rougeL']}")
    print(f"  Accuracy : {report['classification']['accuracy']}")
    print(f"  Macro F1 : {report['classification']['macro_f1']}")
    print(f"  Precision: {report['classification']['precision']}")
    print(f"  Recall   : {report['classification']['recall']}")
    print(f"  Sev R2   : {report['severity']['r2']}")
    print(f"  Sev MAE  : {report['severity']['mae']}")
    print(f"{'='*50}")
    print(f"Saved → {EVAL_OUTPUT}")

if __name__ == "__main__":
    main()

import json
import re
import torch
import nltk
import numpy as np
from PIL import Image
import sklearn.metrics as skm
from scipy.stats import pearsonr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, CLIPModel, CLIPProcessor
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_module

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

ADAPTER_PATH    = "outputs/qwen25vl_plantvillage/best"
BASE_MODEL      = "Qwen/Qwen2.5-VL-7B-Instruct"
CAPTIONS        = "captions.json"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
OUTPUT_REPORT   = "evaluation_report.json"
MAX_EVAL        = 200

EVAL_INSTRUCTION = (
    "Analyze this plant leaf image and provide a detailed diagnosis. "
    "Include: plant name, disease or health condition, severity level, "
    "causal pathogen, and visible symptoms."
)

CONDITION_LABELS = [
    "Healthy", "Late Blight", "Early Blight", "Bacterial Spot",
    "Leaf Mold", "Septoria Leaf Spot", "Spider Mite Damage",
    "Target Spot", "Mosaic Virus", "Yellow Leaf Curl Virus",
]

SEVERITY_SCALE = {
    "none": 0.0, "mild": 1.0, "mild to moderate": 1.5,
    "moderate": 2.0, "moderate to severe": 2.5, "severe": 3.0,
}


def extract_condition(text):
    text_l = text.lower()
    m = re.search(r"condition[:\s]+([^.]+)", text_l)
    if m:
        cand = m.group(1).strip().title()
        for label in CONDITION_LABELS:
            if label.lower() in cand.lower() or cand.lower() in label.lower():
                return label
    for label in CONDITION_LABELS:
        if label.lower() in text_l:
            return label
    return "Unknown"


def extract_severity(text):
    text_l = text.lower()
    for key, val in sorted(SEVERITY_SCALE.items(), key=lambda x: -len(x[0])):
        if key in text_l:
            return val
    return 0.0


def load_model():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
    print("Loading Qwen2.5-VL-7B in bfloat16...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print("VLM ready.\n")
    return model, processor


def load_clip():
    print("Loading CLIP...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to("cuda")
    clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()
    return clip_model, clip_proc


def generate_caption(image, model, processor):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": EVAL_INSTRUCTION},
        ],
    }]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=160, num_beams=4,
            repetition_penalty=2.5, no_repeat_ngram_size=6, early_stopping=True,
        )
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def clip_sim(image, text, clip_model, clip_proc):
    inputs = clip_proc(text=[text], images=image, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out     = clip_model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return (img_emb * txt_emb).sum().item()


def main():
    with open(CAPTIONS) as f:
        val_data = json.load(f)["val"]
    if MAX_EVAL:
        val_data = val_data[:MAX_EVAL]
    print(f"Evaluating on {len(val_data)} validation samples\n")

    model, processor   = load_model()
    clip_model, clip_proc = load_clip()
    rouge_scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)

    refs, hyps          = [], []
    rouge_scores        = []
    clip_scores         = []
    true_conditions     = []
    pred_conditions     = []
    true_severities     = []
    pred_severities     = []

    for i, rec in enumerate(val_data):
        image    = Image.open(rec["image"]).convert("RGB")
        ref_cap  = rec["caption"]
        pred_cap = generate_caption(image, model, processor)

        refs.append([ref_cap.lower().split()])
        hyps.append(pred_cap.lower().split())
        rouge_scores.append(rouge_scorer.score(ref_cap, pred_cap)["rougeL"].fmeasure)
        clip_scores.append(clip_sim(image, pred_cap, clip_model, clip_proc))
        true_conditions.append(extract_condition(ref_cap))
        pred_conditions.append(extract_condition(pred_cap))
        true_severities.append(extract_severity(ref_cap))
        pred_severities.append(extract_severity(pred_cap))

        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(val_data)}] done...")

    smooth  = SmoothingFunction().method1
    bleu1   = corpus_bleu(refs, hyps, weights=(1,0,0,0),         smoothing_function=smooth)
    bleu4   = corpus_bleu(refs, hyps, weights=(.25,.25,.25,.25), smoothing_function=smooth)
    rougeL  = float(np.mean(rouge_scores))
    clip_s  = float(np.mean(clip_scores))

    accuracy    = sum(p == t for p, t in zip(pred_conditions, true_conditions)) / len(true_conditions)
    known_mask  = [t in CONDITION_LABELS for t in true_conditions]
    t_known     = [t for t, k in zip(true_conditions, known_mask) if k]
    p_known     = [p for p, k in zip(pred_conditions, known_mask) if k]
    labels_used = sorted(set(t_known))

    macro_f1  = skm.f1_score(        t_known, p_known, labels=labels_used, average="macro", zero_division=0)
    macro_pre = skm.precision_score(  t_known, p_known, labels=labels_used, average="macro", zero_division=0)
    macro_rec = skm.recall_score(     t_known, p_known, labels=labels_used, average="macro", zero_division=0)

    per_class_f1 = {}
    for label in labels_used:
        y_t = [1 if t == label else 0 for t in t_known]
        y_p = [1 if p == label else 0 for p in p_known]
        per_class_f1[label] = round(skm.f1_score(y_t, y_p, zero_division=0), 4)

    sev_r2  = float("nan")
    sev_mae = float(np.mean(np.abs(np.array(true_severities) - np.array(pred_severities))))
    if len(set(true_severities)) > 1:
        try:
            corr, _ = pearsonr(true_severities, pred_severities)
            sev_r2  = round(float(corr ** 2), 4)
        except Exception:
            pass

    div = "=" * 57
    print(f"\n{div}")
    print(f"  EVALUATION REPORT  Qwen2.5-VL-7B PlantVillage")
    print(f"  Samples: {len(val_data)}")
    print(div)
    print(f"\n  BLEU-1  : {bleu1:.4f}   (target > 0.40)")
    print(f"  BLEU-4  : {bleu4:.4f}   (target > 0.25)")
    print(f"  ROUGE-L : {rougeL:.4f}   (target > 0.40)")
    print(f"  CLIP Sim: {clip_s:.4f}   (target > 0.25)")
    print(f"\n  Accuracy      : {accuracy:.4f}   (target > 0.85)")
    print(f"  Macro F1      : {macro_f1:.4f}   (target > 0.80)")
    print(f"  Precision     : {macro_pre:.4f}")
    print(f"  Recall        : {macro_rec:.4f}")
    print(f"\n  Severity R2   : {sev_r2:.4f}   (target > 0.70)")
    print(f"  Severity MAE  : {sev_mae:.4f}")
    print(f"\n  Per-class F1:")
    for cls, f1 in sorted(per_class_f1.items(), key=lambda x: -x[1]):
        bar = chr(9608) * int(f1 * 25)
        print(f"  {cls:<30} {f1:.4f}  {bar}")
    print(f"\n{div}\n")

    report = {
        "model": "Qwen2.5-VL-7B-Instruct (bfloat16, LoRA r32)",
        "n_eval": len(val_data),
        "caption_quality": {"bleu1": round(bleu1,4), "bleu4": round(bleu4,4), "rougeL": round(rougeL,4), "clip_similarity": round(clip_s,4)},
        "classification": {"accuracy": round(accuracy,4), "macro_f1": round(macro_f1,4), "precision": round(macro_pre,4), "recall": round(macro_rec,4)},
        "severity": {"r2": sev_r2, "mae": round(sev_mae,4)},
        "per_class_f1": per_class_f1,
    }
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()

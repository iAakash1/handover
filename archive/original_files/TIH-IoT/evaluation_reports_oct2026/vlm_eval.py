"""Evaluate a PlantVillage VLM checkpoint with the same protocol as VLM/qwen/04_evaluate.py.

Same 200 validation records (captions.json "val"[:200]), same caption / classification /
severity metrics, plus per-sample predictions so the report can be audited.

Usage: python vlm_eval.py <kind> <out_dir> [adapter_dir]
  kind: llava | blip2 | florence | qwen
"""
import json
import os
import re
import sys
import time

import numpy as np
import sklearn.metrics as skm
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image
from rouge_score import rouge_scorer as rouge_scorer_module
from scipy.stats import pearsonr
from transformers import CLIPModel, CLIPProcessor

KIND, OUT_DIR = sys.argv[1], sys.argv[2]
ADAPTER = sys.argv[3] if len(sys.argv) > 3 else None
CAPTIONS = r"C:\Users\Admin\Desktop\VLM\qwen\captions.json"
MAX_EVAL = int(os.environ.get("MAX_EVAL", "200"))
INSTRUCTION = (
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


def load(kind):
    if kind == "llava":
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        base_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        proc = LlavaNextProcessor.from_pretrained(ADAPTER or base_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="cuda")
    elif kind == "blip2":
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        base_id = "Salesforce/blip2-opt-2.7b"
        proc = Blip2Processor.from_pretrained(ADAPTER or base_id)
        model = Blip2ForConditionalGeneration.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="cuda")
    elif kind == "florence":
        from transformers import AutoModelForCausalLM, AutoProcessor
        base_id = "microsoft/Florence-2-base-ft"
        proc = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.float32, trust_remote_code=True).to("cuda")
    elif kind == "qwen":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        base_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        proc = AutoProcessor.from_pretrained(ADAPTER or base_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="cuda")
    else:
        raise SystemExit(kind)
    if ADAPTER:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    return model, proc, base_id


def generate(kind, model, proc, image, instruction=INSTRUCTION):
    with torch.no_grad():
        if kind in ("llava", "qwen"):
            msgs = [{"role": "user", "content": [
                {"type": "image"} if kind == "llava" else {"type": "image", "image": image},
                {"type": "text", "text": instruction}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[image], return_tensors="pt").to("cuda")
            out = model.generate(**inputs, max_new_tokens=160, num_beams=4,
                                 repetition_penalty=2.5, no_repeat_ngram_size=6, early_stopping=True)
            return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if kind == "blip2":
            inputs = proc(images=image, return_tensors="pt").to("cuda", torch.bfloat16)
            out = model.generate(**inputs, max_new_tokens=80, num_beams=4, repetition_penalty=3.0,
                                 no_repeat_ngram_size=8, length_penalty=0.6, early_stopping=True)
            raw = proc.decode(out[0], skip_special_tokens=True).strip()
            clean = ". ".join(raw.split(". ")[:3]).strip()
            return clean if clean.endswith(".") else clean + "."
        if kind == "florence":
            task = "<MORE_DETAILED_CAPTION>"
            inputs = proc(text=task, images=image, return_tensors="pt").to("cuda")
            out = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                 max_new_tokens=160, num_beams=3)
            return proc.batch_decode(out, skip_special_tokens=True)[0].strip()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    val = json.load(open(CAPTIONS))["val"][:MAX_EVAL]
    model, proc, base_id = load(KIND)
    clip_model = CLIPModel.from_pretrained(r"C:/Users/Admin/AppData/Local/Temp/claude/C--Users-Admin-Desktop-reports/2674d687-2292-479f-ba90-63d322a030b8/scratchpad/clip_local").to("cuda").eval()
    clip_proc = CLIPProcessor.from_pretrained(r"C:/Users/Admin/AppData/Local/Temp/claude/C--Users-Admin-Desktop-reports/2674d687-2292-479f-ba90-63d322a030b8/scratchpad/clip_local")
    scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)

    refs, hyps, rouge, clip_s, tc, pc, ts, ps, rows = [], [], [], [], [], [], [], [], []
    t0 = time.time()
    for i, rec in enumerate(val):
        image = Image.open(rec["image"]).convert("RGB")
        ref = rec["caption"]
        try:
            hyp = generate(KIND, model, proc, image, rec["instruction"] if KIND == "llava" else INSTRUCTION)
        except Exception as e:  # keep going; record the failure
            hyp = f"[generation error: {type(e).__name__}: {e}]"
        refs.append([ref.lower().split()])
        hyps.append(hyp.lower().split())
        rouge.append(scorer.score(ref, hyp)["rougeL"].fmeasure)
        ci = clip_proc(text=[hyp[:300]], images=image, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            o = clip_model(**ci)
        ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
        te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
        clip_s.append((ie * te).sum().item())
        tc.append(extract_condition(ref)); pc.append(extract_condition(hyp))
        ts.append(extract_severity(ref)); ps.append(extract_severity(hyp))
        rows.append({"image": os.path.basename(rec["image"]), "class": rec.get("class_name"),
                     "reference": ref, "generated": hyp, "true_condition": tc[-1], "pred_condition": pc[-1]})
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(val)}] {time.time() - t0:.0f}s", flush=True)

    sm = SmoothingFunction().method1
    known = [t in CONDITION_LABELS for t in tc]
    tk = [t for t, k in zip(tc, known) if k]
    pk = [p for p, k in zip(pc, known) if k]
    labels = sorted(set(tk))
    r2 = None
    if len(set(ts)) > 1 and len(set(ps)) > 1:
        r2 = round(float(pearsonr(ts, ps)[0] ** 2), 4)
    report = {
        "model": f"{base_id}" + (f" + LoRA adapter ({os.path.basename(os.path.dirname(ADAPTER))}/best)" if ADAPTER else " (zero-shot, no adapter)"),
        "protocol": "VLM/qwen/04_evaluate.py protocol: captions.json val[:200], seed-42 90/10 split, "
                    "BLEU (nltk corpus, smoothing 1), ROUGE-L, CLIP ViT-B/32 image-text cosine, "
                    "condition parsed from generated text",
        "n_eval": len(val),
        "evaluated_on": time.strftime("%Y-%m-%d"),
        "seconds_per_image": round((time.time() - t0) / len(val), 3),
        "caption_quality": {
            "bleu1": round(corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=sm), 4),
            "bleu4": round(corpus_bleu(refs, hyps, weights=(.25,) * 4, smoothing_function=sm), 4),
            "rougeL": round(float(np.mean(rouge)), 4),
            "clip_similarity": round(float(np.mean(clip_s)), 4),
        },
        "classification": {
            "accuracy": round(sum(p == t for p, t in zip(pc, tc)) / len(tc), 4),
            "macro_f1": round(skm.f1_score(tk, pk, labels=labels, average="macro", zero_division=0), 4),
            "precision": round(skm.precision_score(tk, pk, labels=labels, average="macro", zero_division=0), 4),
            "recall": round(skm.recall_score(tk, pk, labels=labels, average="macro", zero_division=0), 4),
            "unparsed_predictions": pc.count("Unknown"),
        },
        "severity": {"r2": r2, "mae": round(float(np.mean(np.abs(np.array(ts) - np.array(ps)))), 4)},
        "per_class_f1": {l: round(skm.f1_score([t == l for t in tk], [p == l for p in pk], zero_division=0), 4)
                         for l in labels},
    }
    json.dump(report, open(os.path.join(OUT_DIR, "evaluation_report.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(OUT_DIR, "predictions.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

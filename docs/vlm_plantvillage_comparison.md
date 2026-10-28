# Vision-Language Models on PlantVillage (March – August 2026)

**In short:** instead of a CNN that only outputs a label, we fine-tuned Vision-Language Models (VLMs) to *write a
diagnosis* for a leaf photo: plant, disease, severity, pathogen and visible symptoms. Several VLMs were tried,
and **Qwen2.5-VL-7B-Instruct** worked best.

## Month by month

| Month | What happened | Folder |
|---|---|---|
| Mar 2026 | Read VLM papers. Fine-tuned LLaVA-1.6-Mistral-7B and BLIP-2 OPT-2.7B with LoRA (24–25 Mar). | `../2026-03_March/outputs/` |
| Apr 2026 | Florence-2 attempted (stopped by library incompatibilities). Qwen2.5-VL-7B LoRA fine-tuned (6–7 Apr), with a CLIP check that rejects non-leaf photos and a GradCAM heat-map (13 Apr). | `../2026-04_April/florence/`, `../2026-04_April/qwen/`, `../2026-04_April/other/`, `../2026-04_April/latest_changes/` |
| May 2026 | Pipeline restructured as **PlantDx** (15 PlantVillage classes, Streamlit app). | `../2026-05_May/PlantDx/` |
| Jun 2026 | BLIP-2 re-run for 1 epoch with full evaluation (15 Jun). Expert-checked caption set prepared. | `../2026-06_June/tp/` |
| Aug 2026 | **PlantVLM**: Qwen2.5-VL fine-tuned only on 150 captions written and checked by hand (30 images, 6 diseases). | `../2026-08_August/newpipe/` |

## Results on the same 200 validation images

All rows use the protocol of `../2026-04_April/qwen/04_evaluate.py`, so they can be compared directly.

| Model | Accuracy | Macro-F1 | BLEU-4 | Evaluation report |
|---|---:|---:|---:|---|
| Qwen2.5-VL-7B, LoRA r32 (Apr) | **98.50%** | 0.9827 | 0.4556 | `../2026-04_April/qwen/evaluation_report.json` |
| Qwen2.5-VL-7B, PlantDx 15-class (May) | 98.00% | 0.9777 | 0.3864 | `../2026-05_May/PlantDx/outputs/evaluations/report.json` |
| BLIP-2 OPT-2.7B, LoRA (Mar) | 89.00% | 0.8928 | 0.0865 | `../evaluation_reports_oct2026/blip2_march` |
| LLaVA-1.6-Mistral-7B, LoRA (Mar) | 2.00% | 0.0151 | 0.0006 | `../evaluation_reports_oct2026/llava_v16` |
| Qwen2.5-VL-7B, no fine-tuning | 8.50% | 0.0800 | 0.0018 | `../evaluation_reports_oct2026/qwen_zeroshot` |
| Florence-2-base-ft, no fine-tuning | 0.00% | 0.0000 | 0.0016 | `../evaluation_reports_oct2026/florence_zeroshot` |
| BLIP-2 OPT-2.7B, 1-epoch re-run (Jun) | 89.00% | 0.8948 | 0.0865 | `../evaluation_reports_oct2026/blip2_june` (its own June script recorded 0%: `../2026-06_June/tp/evaluation_report.json`) |

PlantVLM (August) is a different task: it uses 6 validation images and hand-written captions, and the
fine-tuned model named the correct disease for 2 of 6 images (chance is 1 in 6).
Report: `../2026-08_August/newpipe/runs/run_002_fold0/evaluation/val_report.json`.

## Trained models (LoRA adapters, stored with Git LFS)
- `../2026-03_March/outputs/llava_plantvillage/best`, `../2026-03_March/outputs/blip2_plantvillage/best`: March runs
- `../2026-04_April/qwen/outputs/qwen25vl_plantvillage/best`: April Qwen2.5-VL
- `../2026-05_May/PlantDx/outputs/checkpoints/best`: May PlantDx
- `../2026-06_June/tp/outputs/blip2_plantvillage/best`: June BLIP-2 re-run
- `../2026-08_August/newpipe/runs/run_002_fold0/checkpoints/best`: August PlantVLM

Load an adapter on top of its Hugging Face base model with `peft.PeftModel.from_pretrained(base, adapter_dir)`.
The training captions (`../2026-04_April/qwen/captions.json`, `../2026-05_May/PlantDx/data/captions/captions.json`) and split
(`../2026-05_May/PlantDx/data/splits/splits.json`) are included; the PlantVillage images are not.

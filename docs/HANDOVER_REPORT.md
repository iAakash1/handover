# Handover report — model by model

Chanakya Fellowship, TIH IIT Bombay · 1 December 2025 – 30 September 2026  
Aakash Trimbak Jawle, Kamaaksha Rajendra Kharul · mentor Dr. Santwana Gudadhe

This report covers every model and experiment we ran: the data, how it was trained, what it scored and where
its code, weights and evaluation reports are. Run commands are in each project README and in `HOW_TO_RUN.md`.
All numbers below come from the evaluation reports in this repository (`evaluation/` and each project folder).

## Timeline

| Month | Work | Project |
|---|---|---|
| Dec 2025 | TIH foundation tasks (image ops, preprocessing, quality checks, CNN basics); custom CNN | 01 |
| Jan 2026 | Keras transfer learning, PyTorch ConvNeXt, crop validation; SoyNet soybean on the VDI | 01, 02, 03 |
| Feb 2026 | PoCRA video editing; AI for Agri conference | – |
| Mar 2026 | VLM papers studied; LLaVA-1.6 and BLIP-2 fine-tuned | 04 |
| Apr 2026 | Florence-2 attempt; Qwen2.5-VL fine-tuned; production app with CLIP check and GradCAM | 05, 06 |
| May 2026 | PlantDx, 15 PlantVillage classes | 07 |
| Jun 2026 | PlantDx published; BLIP-2 re-run; seven-VLM zero-shot benchmark (PoCRA report) | 04 |
| Jul 2026 | Knowledge-grounded PlantDx on Apple MLX, tomato and mango | 09 |
| Aug 2026 | PlantVLM on expert captions; TIH dataset received; SigLIP/OpenCLIP prototype | 08, 11 |
| Sep 2026 | TIH tomato QLoRA runs and cross-domain benchmark → ICAIN 2026 paper (accepted 1 Oct) | 10 |

---

## 01 · CNNs in Keras — tomato leaves (Dec 2025 – Jan 2026)

- **Data:** PlantVillage, 10 tomato classes (9 diseases + healthy).
- **Models:** custom CNN (3 conv blocks, 32×32 input, Adam 1e-4, early stopping); MobileNetV2, EfficientNetB3,
  ConvNeXt-Tiny from ImageNet (224×224, head 10 epochs at 1e-3, then top 20–30 layers 10 epochs at 1e-5,
  flips/rotation/zoom, 80/20 `validation_split`).
- **Also built:** the Week-2 `preprocessing` package and an image-quality gate (blur, brightness, contrast,
  noise, resolution) and a crop check (green-pixel ratio + MobileNetV2 embedding similarity ≥ 0.45).
- **Results (held-out 20% validation, 3,198 images):** ConvNeXt 96.15% (macro-F1 0.9507), EfficientNetB3
  93.59% (0.9275), MobileNetV2 90.90% (0.8939). Custom CNN 98.51% on the tomato test split — indicative only,
  those images overlap its training data.
- **Where:** `projects/01_cnn_keras_tomato` · reports `evaluation/keras/`.

## 02 · ConvNeXt-Tiny in PyTorch (8 Jan 2026) — best CNN

- **Data:** the 10 tomato classes split 70/15/15 (11,203 / 2,397 / 2,411); list in `splits/split_70_15_15.csv`.
- **Training:** full fine-tune from ImageNet, AdamW 1e-4, AMP, batch 64, 10 epochs.
- **Results:** validation 99.79%, **test 99.92%** (macro-F1 0.9991) — 2 of 2,411 test images wrong.
- **Where:** `projects/02_cnn_pytorch_convnext` · report `evaluation/convnext_torch/`.

## 03 · SoyNet — soybean healthy vs diseased (27 Jan 2026)

- **Data:** provided SoyNet camera photos, 256×256: 3,164 diseased, 556 healthy; 70/15/15, seed 42.
- **Training:** provided pipeline on the VDI; Keras run and PyTorch run, ImageNet backbones, Adam 1e-4 → 1e-5.
- **Results (positive class = Healthy):** Keras ResNet50 **95.16%** (ROC-AUC 0.9903); PyTorch EfficientNetB3
  93.93%, EfficientNetB0 93.39%, MobileNetV2 92.50%, ResNet50 91.61% (all with Healthy recall 1.0).
- **Where:** `projects/03_soybean_soynet` (metrics in `results/experiments/<model>/`).

## 04 · LLaVA-1.6 and BLIP-2 with LoRA (Mar, Jun 2026)

- **Task (all VLMs from here on):** write a structured diagnosis (plant, condition, severity, pathogen, symptoms)
  for a PlantVillage leaf; condition is parsed from the text and scored on the same 200 validation images.
- **Results:** BLIP-2 OPT-2.7B + LoRA **89.00%** (March) and 89.00% (June 1-epoch re-run; its own June script had
  reported 0% because of a scoring bug); LLaVA-1.6-Mistral-7B + LoRA r64 2.00% (generic answers). LLaVA-1.5
  stopped at a batch-collation error.
- **Where:** `projects/04_llava_blip2` · reports `evaluation/blip2_march`, `blip2_june`, `llava_v16`.

## 05 · Florence-2 (Apr, Jun 2026)

- Fine-tuning failed on five successive incompatibilities with transformers 5.3. Zero-shot with transformers
  4.46.3: 0% (scene captions, no disease names). Dropped.
- **Where:** `projects/05_florence2` · report `evaluation/florence_zeroshot`.

## 06 · Qwen2.5-VL-7B + LoRA (Apr 2026) — the VLM that worked

- **Training:** LoRA r32 / alpha 64 on all attention and MLP projections, bf16, 3 epochs, LR 2e-4, batch 4 ×
  grad-accum 4, 90/10 split (seed 42).
- **Results:** **98.50%** accuracy, macro-F1 0.9827, BLEU-4 0.4556, ROUGE-L 0.5869, severity R² 0.9654.
  The same model without fine-tuning: 8.50%.
- **App:** `crop_disease_app` — CLIP ViT-B/32 rejects non-leaf photos (threshold 0.27), output parser,
  plant-mismatch check, GradCAM heat-map, Streamlit UI.
- **Where:** `projects/06_qwen25vl_plantvillage` · report `qwen/evaluation_report.json`.

## 07 · PlantDx — 15 PlantVillage classes (May 2026)

- The April pipeline rebuilt as a modular project (`core/`, `utils/`, `scripts/`, `configs/`), trained on 15
  classes (tomato, potato, pepper), Qwen2.5-VL + LoRA r32.
- **Results:** **98.00%**, macro-F1 0.9777, BLEU-4 0.386, severity R² 0.966.
- **Where:** `projects/07_plantdx_plantvillage` · report `outputs/evaluations/report.json`.

## 08 · PlantVLM — expert-checked captions only (Aug 2026)

- Qwen2.5-VL + LoRA r16 on 150 hand-written, expert-revised captions for 30 images (6 diseases).
- **Results:** best epoch 3; right disease for 2 of 6 validation images (chance 1 of 6). Conclusion: good captions
  need far more images.
- **Where:** `projects/08_plantvlm_expert_captions` · report `runs/run_002_fold0/evaluation/val_report.json`.

## 09 · Knowledge-grounded PlantDx on Apple MLX (Jul 2026) — not runnable on this machine

- Captions compiled from a cited Disease Knowledge Base (ontology → vocabulary → templates, 1,966 captions);
  Qwen2.5-VL-7B-Instruct-4bit + QLoRA r16 with `mlx-vlm` on an Apple M4 Pro (24 GB).
- **Results:** tomato 7.3% → **93.7%** (macro-F1 0.919, 910 test images); mango 17.5% → **82.0%** (0.811, 200).
- MLX = Apple's framework for Apple Silicon; the adapters stayed on the Mac. Results, configs and Mac commands:
  `projects/09_plantdx_mlx_knowledge_grounded`.

## 10 · TIH tomato, 11 classes, Qwen2.5-VL QLoRA (Aug – Sep 2026) — ICAIN 2026 paper

- **Data:** private TIH `Tomato_12thAug26`, 11 classes, 7,095 images → 6,545 after SHA-256 de-duplication and
  removal of contradictory labels; 80/10/10 stratified split (5,247 / 649 / 649), seed 20260911.
- **Training:** 4-bit NF4 QLoRA, LoRA r64 / alpha 128 on q/k/v/o, vision tower frozen, paged AdamW 8-bit, LR 4e-5,
  532 px, micro-batch 1 × grad-accum 8, class-balanced sampling, answer-only loss; 2.9 epochs, 193 min, 10.6 GB.
- **Results:** test **92.76%**, macro-F1 0.9378 (0.327 s/image); earlier 'strong' run 87.98%. Cross-domain set
  of 176 images from another source: **69.32%** (macro-F1 0.7195); Late Blight is the weak class there (F1 14.29%).
- **Where:** `projects/10_tih_tomato_qwen_qlora` (`reports/`, `checkpoints/`, `RESULTS.md`).

## 11 · SigLIP / OpenCLIP prototype (Aug 2026)

- Contrastive image–text classification pipeline for the TIH data; written, not trained (Qwen QLoRA was chosen).
- **Where:** `projects/11_siglip_openclip_prototype`.

---

## Evaluation protocol in one place

| Family | Data | Metrics | Script |
|---|---|---|---|
| CNNs (01, 02) | original held-out split | accuracy, macro / weighted P-R-F1, confusion matrix | `evaluate.py` in each project |
| PlantVillage VLMs (04–07) | the same 200 validation images | condition accuracy and F1 parsed from the text, BLEU, ROUGE-L, CLIP similarity, severity | `evaluation/vlm_eval.py` |
| TIH tomato (10) | frozen 649-image test + 176-image external set | accuracy, macro / weighted F1, per-class, latency | `plantdx.cli company evaluate / evaluate-external` |

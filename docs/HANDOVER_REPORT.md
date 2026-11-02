# Handover report: crop disease detection with CNNs and Vision-Language Models

TIH IIT Bombay · Chanakya Fellowship (Agri-AI 2025) · 1 December 2025 – 30 September 2026
Aakash Trimbak Jawle, Kamaaksha Rajendra Kharul · Faculty mentor: Dr. Santwana Gudadhe (PCCOE Pune) ·
TIH mentors: Harish Lokawad, Hrishikesh Patil, Rutuja Chavan

The central, model-wise report of the handover. Every number comes from an evaluation report in this repository;
its path is given next to it. Commands for each model are in its `HANDOVER_GUIDE.md` (one to two pages) and in
[`HOW_TO_RUN.md`](HOW_TO_RUN.md). Where to start: [`README_HANDOVER.md`](README_HANDOVER.md).

**Contents:** [1 Overview](#1-project-overview-and-objectives) · [2 Model inventory](#2-model-inventory) ·
[3 Software and hardware](#3-software-and-hardware) · [4 Models](#4-models) ·
[5 Evaluation methodology](#5-evaluation-methodology) · [6 Datasets](#6-datasets) ·
[7 Research papers](#7-research-papers) · [8 Limitations and unsuccessful approaches](#8-limitations-and-unsuccessful-approaches) ·
[9 Reproduction](#9-reproduction) · [10 Future development](#10-future-development)

---

## 1. Project overview and objectives

The fellowship track was *image-based disease detection for crops using AI/ML and computer vision*. The work went
in four steps:

1. **December 2025 – January 2026:** TIH foundation tasks (image operations, preprocessing, quality checks) and CNN
   classifiers for tomato leaves (PlantVillage) and soybean (SoyNet).
2. **March – May 2026:** vision-language models (VLMs) that write a structured diagnosis instead of a label; LoRA
   fine-tuning of BLIP-2, LLaVA and Qwen2.5-VL on PlantVillage, and a production app with an out-of-distribution check.
3. **June – July 2026 (Apple Mac, MLX):** knowledge-grounded captions compiled from a cited disease knowledge base,
   QLoRA fine-tuning for tomato and mango.
4. **August – September 2026:** the private 11-class TIH tomato dataset: 4-bit QLoRA of Qwen2.5-VL, a frozen
   test split and a cross-domain benchmark, published at ICAIN 2026.

## 2. Model inventory

| # | Model | Task | Weights | Result (report) |
|---|---|---|---|---|
| 01 | Custom CNN, MobileNetV2, EfficientNetB3, ConvNeXt-Tiny (Keras) | tomato, 10 classes | `projects/01_cnn_keras_tomato/model/*.keras` | ConvNeXt 96.15% validation ([`evaluation/keras/`](../evaluation/keras/README.md)) |
| 02 | ConvNeXt-Tiny (PyTorch) | tomato, 10 classes | `projects/02_cnn_pytorch_convnext/models/convnext_tomato_best.pth` | **99.92% test** ([`evaluation/convnext_torch/`](../evaluation/convnext_torch/README.md)) |
| 03 | ResNet50, EfficientNetB0/B3, MobileNetV2 | soybean, healthy / diseased | `projects/03_soybean_soynet/results/experiments/<model>/model.pth`, `saved_models/` | ResNet50 (Keras) 95.16% test (`results/experiments/`) |
| 04 | BLIP-2 OPT-2.7B + LoRA; LLaVA-1.6 + LoRA | structured diagnosis, PlantVillage | `projects/04_llava_blip2/outputs/*/best` | BLIP-2 89.00%; LLaVA 2.00% (`evaluation/blip2_march`, `llava_v16`) |
| 06 | Qwen2.5-VL-7B + LoRA, crop disease app | structured diagnosis, PlantVillage | `projects/06_qwen25vl_plantvillage/qwen/outputs/qwen25vl_plantvillage/best` | **98.50%** (`qwen/evaluation_report.json`) |
| 07 | PlantDx: Qwen2.5-VL-7B + LoRA | 15 PlantVillage classes | `projects/07_plantdx_plantvillage/outputs/checkpoints/best` | 98.00% (`outputs/evaluations/report.json`) |
| 08 | PlantVLM: Qwen2.5-VL-7B + LoRA on expert captions | tomato captions, 30 images | `projects/08_plantvlm_expert_captions/runs/run_002_fold0/checkpoints/best` | 2 of 6 (exploratory; `runs/run_002_fold0/evaluation/val_report.json`) |
| 09A | Qwen2.5-VL-7B 4-bit + QLoRA on **MLX**, tomato | tomato captions, 10 classes | `projects/09_plantdx_mlx_knowledge_grounded/checkpoints/qwen25vl_tomato_qlora/adapters.safetensors` (Nov 2026) | July 93.7%; Nov 90.9% (section 4.8) |
| 09B | same, mango | mango captions, 8 classes | `…/checkpoints/qwen25vl_mango_qlora/adapters.safetensors` (Nov 2026) | July 82.0%; Nov 86.0% (section 4.8) |
| 10 | Qwen2.5-VL-7B 4-bit NF4 QLoRA | TIH tomato, 11 classes | `projects/10_tih_tomato_qwen_qlora/checkpoints/tih_tomato_moreaccurate` | **92.76% test, 69.32% external** (`reports/`) |

All weights are kept out of git and fetched from github.com/iAakash1/TIH-IoT (Git LFS) by
`python setup\download_models.py` (final models, ~4.0 GB; `--all` adds older versions and per-epoch checkpoints,
+3.6 GB), each checked against the SHA-256 in [`setup/models_manifest.json`](../setup/models_manifest.json).
Project 00 holds the December 2025 TIH notebooks (coursework, no model).

## 3. Software and hardware

| Environment | Main packages | Hardware | Projects |
|---|---|---|---|
| `keras` | TensorFlow 2.20, Keras 3.12 | CPU is enough | 01 |
| `soynet` | TensorFlow 2.10, PyTorch 2.5.1 (CUDA 12.1) | NVIDIA GPU | 03 |
| `vlm` | PyTorch 2.5.1 (CUDA 12.1), transformers 5.3, PEFT 0.18, bitsandbytes 0.49 | NVIDIA GPU, 8–24 GB | 02, 04, 06, 07, 08 |
| `florence` | as `vlm`, transformers 4.46.3 | NVIDIA GPU | Florence-2 (failed experiment) |
| `tih` | PyTorch 2.7.1 (CUDA 11.8), transformers 5.3, PEFT 0.18, bitsandbytes 0.49 | NVIDIA GPU, 24 GB | 10 |
| MLX (Mac) | Python 3.12, `mlx` 0.31.2, `mlx-vlm` 0.6.4, `mlx-lm` 0.31.3, transformers 5.12.1 | **Apple Silicon** (trained on an M4 Pro, 24 GB) | 09 |

The Windows environments are built by `python setup\setup_envs.py` from the pinned files in `setup/requirements/`
(Python 3.10); the TIH workstation has an NVIDIA RTX 4500 Ada (24 GB). Project 09 uses its own environment
(`projects/09_…/requirements-mlx.txt`) and does not run on CUDA; projects 06, 07, 08 and 10 do not run on MLX.

---

## 4. Models

### 4.1 Project 01: tomato CNNs in Keras · [guide](../projects/01_cnn_keras_tomato/HANDOVER_GUIDE.md)

- **Method and architecture.** Custom CNN (three Conv-BatchNorm-MaxPool blocks of 32/64/128 filters, Dense 128,
  dropout 0.5, 32×32 input) and transfer learning with ImageNet MobileNetV2, EfficientNetB3 and ConvNeXt-Tiny
  (224×224 input; classifier head 10 epochs at LR 1e-3, then the top 20–30 backbone layers 10 epochs at LR 1e-5;
  flips/rotation/zoom; early stopping). Optional inference gates: image quality (blur, brightness, contrast, noise,
  resolution; `preprocessing/quality_checks.py`) and a crop check (green and edge ratio, then MobileNetV2
  embedding similarity; `validation/`).
- **Data.** PlantVillage, 10 tomato classes; Keras `validation_split` 80/20 (3,198 validation images).
- **Results** ([`evaluation/keras/`](../evaluation/keras/README.md)): ConvNeXt-Tiny 96.15% (macro-F1 0.9507), EfficientNetB3 93.59% (0.9275),
  MobileNetV2 90.90% (0.8939); custom CNN 98.51% on the tomato test split, which overlaps its training data.

### 4.2 Project 02: ConvNeXt-Tiny in PyTorch · [guide](../projects/02_cnn_pytorch_convnext/HANDOVER_GUIDE.md)

- **Method.** torchvision ConvNeXt-Tiny from ImageNet, fully fine-tuned: AdamW 1e-4, mixed precision, batch 64,
  10 epochs.
- **Data.** 10 tomato classes, 70/15/15 per class: 11,203 / 2,397 / 2,411 (`splits/split_70_15_15.csv`).
- **Results** ([`evaluation/convnext_torch/`](../evaluation/convnext_torch/README.md)): validation 99.79%, **test 99.92%**, macro-F1 0.9991.

### 4.3 Project 03: SoyNet soybean classifiers · [guide](../projects/03_soybean_soynet/HANDOVER_GUIDE.md)

- **Method.** The SoyNet pipeline run with Keras (ResNet50, MobileNetV2) and PyTorch (ResNet50, EfficientNetB0/B3,
  MobileNetV2): ImageNet weights, batch 32, up to 50 epochs, Adam 1e-4 then 1e-5, early stopping (patience 15).
- **Data.** SoyNet camera photos, 256×256: 3,164 diseased, 556 healthy; stratified 70/15/15, seed 42 (2,603 / 557 /
  560; `splits/soynet_pytorch_split_70_15_15_seed42.csv`).
- **Results** (positive class = Healthy; `results/experiments/<model>/metrics.json`): Keras ResNet50 **95.16%**
  (ROC-AUC 0.9903), MobileNetV2 89.07%; PyTorch EfficientNetB3 93.93%, EfficientNetB0 93.39%, MobileNetV2 92.50%,
  ResNet50 91.61%. A further soybean model trained on the TIH VDI stayed on the VDI.

### 4.4 Project 04: BLIP-2 and LLaVA with LoRA · [guide](../projects/04_llava_blip2/HANDOVER_GUIDE.md)

- **Method.** BLIP-2 OPT-2.7B with LoRA r32 learns to write a structured diagnosis; March run 5 epochs (5,800
  steps), June a one-epoch re-run. LLaVA-1.6-Mistral-7B with LoRA r64 (1,700 steps) for comparison.
- **Results** (shared 200-image set, section 5): BLIP-2 March **89.00%** (macro-F1 0.8928, `evaluation/blip2_march/`),
  June 89.00% (0.8948, `evaluation/blip2_june/`); LLaVA-1.6 2.00% (`evaluation/llava_v16/`).

### 4.5 Project 06: Qwen2.5-VL-7B with LoRA, crop disease app · [guide](../projects/06_qwen25vl_plantvillage/HANDOVER_GUIDE.md)

- **Method.** `Qwen2.5-VL-7B-Instruct`, bf16, LoRA r32 / alpha 64 on all attention and MLP projections, 3 epochs,
  LR 2e-4, batch 4 × accumulation 4. App pipeline: CLIP ViT-B/32 out-of-distribution check (stop below 0.27) →
  generation (beam search 4) → field parser → plant-mismatch check → GradCAM; the displayed confidence comes from
  the CLIP similarity.
- **Data.** PlantVillage, 15 classes, captions in `qwen/captions.json` (90/10, seed 42; 18,574 / 2,064).
- **Results** (`qwen/evaluation_report.json`): **98.50%**, macro-F1 0.9827, BLEU-4 0.4556, ROUGE-L 0.5869,
  severity R² 0.9654; without fine-tuning 8.50% (`evaluation/qwen_zeroshot/`).

### 4.6 Project 07: PlantDx, 15 PlantVillage classes · [guide](../projects/07_plantdx_plantvillage/HANDOVER_GUIDE.md)

- **Method.** The project-06 recipe as a modular code base (`core/`, `utils/`, `scripts/`, `configs/config.yaml`):
  LoRA r32 / alpha 64, 3 epochs, batch 4 × 4, LR 2e-4.
- **Results** (`outputs/evaluations/report.json`): **98.00%**, macro-F1 0.9777, BLEU-4 0.386, severity R² 0.966.

### 4.7 Project 08: PlantVLM on expert captions · [guide](../projects/08_plantvlm_expert_captions/HANDOVER_GUIDE.md)

- **Method.** Qwen2.5-VL-7B, LoRA r16 / alpha 32 on the language model, up to 8 epochs with early stopping, LR 1e-4.
- **Data.** 30 tomato images (6 diseases × 5) with 150 expert-revised captions; 3 / 1 / 1 images per class.
- **Results** (`runs/run_002_fold0/evaluation/val_report.json`): best epoch 3; right disease for 2 of 6 validation
  images (chance 1 of 6); BLEU-4 0.045. Exploratory: the dataset is too small.

### 4.8 Project 09: knowledge-grounded PlantDx on Apple MLX · [guide](../projects/09_plantdx_mlx_knowledge_grounded/HANDOVER_GUIDE.md)

- **Method.** Captions compiled from a cited Disease Knowledge Base (ontology → vocabulary → concept models →
  templates → 1,966 validated captions; deterministic and content-hashed). `mlx-community/Qwen2.5-VL-7B-Instruct-4bit`
  with QLoRA via `mlx_vlm.lora`: rank 16, alpha 32, dropout 0.05 on q/k/v/o and gate/up/down of all 28
  language-model layers (196 modules, 40,370,176 trainable parameters), vision tower frozen, LR 1e-4,
  micro-batch 1 × accumulation 8, 1 epoch, 448×448 images, answer-only loss.
- **Data.** Tomato: PlantVillage, 10 classes, 18,006 `*.JPG` images, 16,201 / 895 / 910. Mango: MangoLeafBD, 8
  classes, 4,000 images, 3,600 / 200 / 200. Seed 20260711; both image sets are in the project folder.
- **Weights and results.** The July adapters were no longer on the Mac; both models were retrained on the same Mac
  with the same code, data and command (31 Oct – 1 Nov 2026). The **served adapters are the November ones**. Test
  results, with the adapter that produced them:

| Model | Adapter | Test accuracy | Macro-F1 | Report |
|---|---|---:|---:|---|
| Tomato (910 images) | base model, no adapter | 7.3% | 0.034 | `reports/qwen25vl_tomato_qlora/evaluation/metrics.json` |
| | July 2026 (weights not available) | 93.7% | 0.919 | same file |
| | **Nov 2026, iteration 13,200 (served)** | **90.9%** | **0.877** | `retrained_2026-11/qwen25vl_tomato_qlora/evaluation/` |
| | Nov 2026, final iteration 16,201 | 86.9% | 0.847 | `retrained_2026-11/qwen25vl_tomato_qlora/final_iteration_16201/evaluation/` |
| Mango (200 images) | base model, no adapter | 17.5% | 0.128 | `reports/qwen25vl_mango_qlora/evaluation/metrics.json` |
| | July 2026 (weights not available) | 82.0% | 0.811 | same file |
| | **Nov 2026, final iteration 3,600 (served)** | **86.0%** | **0.891** | `retrained_2026-11/qwen25vl_mango_qlora/evaluation/` |

  The tomato snapshot was chosen on the validation split (iteration 13,200: 92.6%, macro-F1 0.904;
  `retrained_2026-11/qwen25vl_tomato_qlora/snapshot_selection/validation_scores.csv`) before it was run on the test
  set. Paths are relative to `projects/09_plantdx_mlx_knowledge_grounded/`.

### 4.9 Project 10: TIH tomato, 11 classes, Qwen2.5-VL QLoRA · [guide](../projects/10_tih_tomato_qwen_qlora/HANDOVER_GUIDE.md)

- **Method.** `Qwen2.5-VL-7B-Instruct` in 4-bit NF4 (double quantisation, bf16 compute) with PEFT LoRA r64 / alpha 128
  on q/k/v/o, vision tower frozen, paged AdamW 8-bit, LR 4e-5, 532 px longest side, micro-batch 1 × accumulation 8,
  class-balanced sampling, answer-only loss; best checkpoint by validation macro-F1 (checkpoint-1908, epoch 2.909;
  193 min, 10.6 GB). The earlier "strong" run: r32 / alpha 64, LR 5e-5, 392 px.
- **Inference.** Greedy generation of a class name; `parse_prediction` maps the answer to a class (exact name, then
  the longest class name contained in it); confidence is the mean probability of the generated tokens (the app
  flags results below 0.55). Streamlit app with diagnose, training, adapter and evaluation pages.
- **Data.** Private TIH `Tomato_12thAug26`: 7,095 images → 6,545 after integrity checks, SHA-256 de-duplication and
  removal of contradictory labels; stratified 80/10/10, seed 20260911 (5,247 / 649 / 649); 176 external images.
- **Results.** Internal test **92.76%**, macro-F1 0.9378, 0.327 s per image
  (`reports/tih_tomato_moreaccurate/test_finetuned/metrics.json`); external **69.32%**, macro-F1 0.7195, late blight
  F1 14.29% (`reports/tih_tomato_moreaccurate_external/`); "strong" run 87.98% (`reports/tih_tomato_qwen_qlora_strong/`).
  These are the numbers of the ICAIN 2026 paper (accuracy 92.76%, macro-F1 93.78%; external 69.32%, 71.95%).

---

## 5. Evaluation methodology

| Models | Test data | Metrics | Script |
|---|---|---|---|
| CNNs (01, 02) | the original held-out split | accuracy, macro / weighted P-R-F1, confusion matrix | `evaluate.py` in each project |
| SoyNet (03) | 70/15/15 test split | accuracy, Healthy-class P-R-F1, ROC-AUC | run at the end of training |
| PlantVillage VLMs (04, 06, 07; Florence-2 and base Qwen) | the same 200 validation images | condition parsed from the text → accuracy, F1; BLEU, ROUGE-L, CLIP similarity, severity | [`evaluation/vlm_eval.py`](../evaluation/README.md) |
| MLX PlantDx (09) | 910 tomato / 200 mango test images, base vs fine-tuned | accuracy, macro / weighted F1, BLEU-4, ROUGE-L, METEOR, CIDEr, BERTScore, hallucination, latency | `plantdx evaluate` |
| TIH tomato (10) | frozen 649-image test set; 176 external images | accuracy, macro / weighted F1, per class, latency | `plantdx.cli company evaluate / evaluate-external` |

Validation results (model selection) and test results are reported separately throughout. Shared 200-image
comparison: Qwen2.5-VL LoRA 98.50%, PlantDx 98.00%, BLIP-2 89.00%, Qwen2.5-VL without fine-tuning 8.50%, LLaVA-1.6
2.00%, Florence-2 zero-shot 0.00% ([`vlm_plantvillage_comparison.md`](vlm_plantvillage_comparison.md)).

## 6. Datasets

Full register with sources, counts, splits, paths and access: [`datasets.md`](datasets.md).

| Dataset | Access | Used by |
|---|---|---|
| PlantVillage (tomato, potato, pepper) | public; images not in git except project 09's tomato copy | 01, 02, 04, 06, 07, 08, 09 |
| MangoLeafBD (CC BY 4.0) | public; in git (project 09) | 09 |
| SoyNet camera photos | provided dataset; not in git | 03 |
| TIH `Tomato_12thAug26` | **private**; on the TIH workstation only, access through TIH-IoT | 10 |

## 7. Research papers

Register: [`publication.md`](publication.md); copies in [`publication/`](publication/).

| Paper | Venue and status | Work |
|---|---|---|
| *Efficient QLoRA Fine-Tuning of Qwen2.5-VL for Multi-Class Tomato Disease Classification and Cross-Domain Evaluation* (ID 953) | **ICAIN 2026**, Proceedings in Springer LNNS: **accepted** 1 Oct 2026; registration and fees completed 21 Oct 2026 · [camera-ready PDF](publication/ICAIN2026_paper953_camera_ready.pdf) | project 10 |
| *PlantDx: Knowledge-Grounded Instruction Tuning for Vision-Language Crop Disease Diagnosis* | draft of 21 Aug 2026; no submission recorded | project 09 |
| *A Vision-Language Framework for Structured Plant Disease Diagnosis with Out-of-Distribution Detection* | draft of 31 Aug 2026; no submission recorded | project 06 |

## 8. Limitations and unsuccessful approaches

**Limitations of the final models.** The PlantVillage models (01, 02, 04, 06, 07, 09) are trained and tested on lab
photos with plain backgrounds; field performance was not measured. The TIH model loses 23 points on photos from
another source (69.32%). The 4-bit 7B models need a 24 GB NVIDIA GPU (06, 07, 08, 10) or an Apple Silicon Mac (09).
MLX retraining is not repeatable bit for bit (mlx-vlm 0.6.4 takes no seed).

**Unsuccessful approaches** ([`failed_experiments/`](../failed_experiments/README.md)):

| Experiment | Outcome |
|---|---|
| Florence-2 fine-tuning | never finished (incompatible with transformers 5.3); 0% zero-shot |
| LLaVA-1.6-Mistral-7B + LoRA | 2.00%, generic answers |
| LLaVA-1.5-7B | stopped by a batch-collation error |
| BLIP-2 June re-run | its own script reported 0% (a scoring problem; 89.00% with the shared protocol) |
| First Qwen2.5-VL run | crashed until four fixes (`fix_*.py`) |
| Zero-shot VLMs | 0–23% |
| PlantVLM on 30 images | 2 of 6 |
| Automatic TIH training run | stopped at step ~40; only a zero-shot baseline (8.47%) |
| SigLIP / OpenCLIP prototype | written, not trained |
| Keras EfficientNet on SoyNet | models did not build |
| Earlier Mac approaches (PlantCaption, caption sets, 4- and 7-VLM MLX benchmarks) | led to the knowledge-grounded pipeline of project 09 |

Superseded versions of the final projects are in [`archive/previous_versions/`](../archive/previous_versions/).

## 9. Reproduction

1. Clone the repository; build the environments (`python setup\setup_envs.py`), fetch the weights
   (`python setup\download_models.py`), check (`python setup\check_setup.py`, `python setup\verify_repository.py --with-projects`).
2. Put the datasets where [`datasets.md`](datasets.md) says, or set the environment variables listed there.
3. Follow the model's `HANDOVER_GUIDE.md`, sections 7 (inference) and 8 (evaluation). Each guide gives the exact
   command and the report it reproduces.
4. Project 09 on a Mac: [`projects/09_…/docs/REPRODUCTION.md`](../projects/09_plantdx_mlx_knowledge_grounded/docs/REPRODUCTION.md). Its training datasets regenerate
   row for row from the images in git (`plantdx normalize` + `prepare-training`).

## 10. Future development

- **Field robustness:** add field photos from varied devices and backgrounds; keep the 176-image external set as a check.
- **Confusable classes:** late blight vs target leaf spot (TIH), early blight vs late blight (MLX tomato), mango die back.
- **Out-of-distribution screening:** the CLIP check of projects 06 and 07 can be placed in front of the 09 and 10 models.
- **Repeatable MLX training:** seed the example order and LoRA initialisation in `mlx_vlm.lora`.
- **More crops:** project 09 adds a crop with a knowledge-base entry, a train config and images; project 10 with a
  class profile.
- **Deployment:** smaller backbones or distillation for cheaper serving than a 24 GB GPU.

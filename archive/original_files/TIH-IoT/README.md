# TIH-IoT: Crop Disease Detection, Chanakya Fellowship (TIH IIT Bombay)

Image-based crop disease detection with CNNs and Vision-Language Models for **tomato, mango and soybean**,
carried out from **December 2025 to September 2026** under the TIH IIT Bombay Chanakya Fellowship (Agri-AI 2025).

**Team:** Kamaaksha Rajendra Kharul, Aakash Trimbak Jawle, Dr. Santwana Gudadhe (Faculty Mentor, PCCOE Pune).
**TIH mentors:** Harish Lokawad, Hrishikesh Patil, Rutuja.

This repository holds everything produced during the fellowship: code, trained models (via Git LFS), evaluation
reports, split files and documents. Datasets are **not** included (PlantVillage, MangoLeafBD, SoyNet and the
TIH tomato dataset have their own licences or owners).

```bash
git lfs install
git clone https://github.com/iAakash1/TIH-IoT.git
```

## Timeline

| Month | Work | Folder |
|---|---|---|
| Dec 2025 | TIH foundation tasks: image operations (package vs scratch), training/inference preprocessing, quality guardrails, CNN theory; custom CNN trained from scratch (29 Dec) | [2025-12_December](2025-12_December) |
| Jan 2026 | Keras transfer learning (MobileNetV2, EfficientNetB3, ConvNeXt-Tiny), PyTorch ConvNeXt-Tiny, crop validation; SoyNet soybean (Keras + PyTorch) on the VDI | [2026-01_January](2026-01_January) |
| Feb 2026 | Video editing for the PoCRA project; attended the AI for Agri conference | [2026-02_February](2026-02_February) |
| Mar 2026 | VLM literature study; LLaVA-1.6 and BLIP-2 LoRA fine-tuning on PlantVillage | [2026-03_March](2026-03_March) |
| Apr 2026 | Research papers (below); Florence-2 attempt; Qwen2.5-VL-7B LoRA (98.50%); CLIP OOD + GradCAM pipeline | [2026-04_April](2026-04_April) |
| May 2026 | PlantDx restructured pipeline, 15 PlantVillage classes (98.00%) | [2026-05_May](2026-05_May) |
| Jun 2026 | PlantDx code published; BLIP-2 re-run; expert-validated caption set; 7-VLM zero-shot benchmark | [2026-06_June](2026-06_June) |
| Jul 2026 | Knowledge-grounded PlantDx with MLX QLoRA: tomato 93.7%, mango 82.0% | [2026-07_July](2026-07_July) |
| Aug 2026 | PlantVLM on 150 expert captions; TIH tomato dataset received; SigLIP/OpenCLIP prototype | [2026-08_August](2026-08_August) |
| Sep 2026 | TIH private tomato dataset (11 classes): Qwen2.5-VL QLoRA runs (92.76% test); cross-domain benchmark (69.32%) | [2026-09_September](2026-09_September) |
| Oct 2026 | 1 Oct: ICAIN 2026 paper accepted; 21 Oct: registration and fees completed. Every saved model re-evaluated with its original split | [evaluation_reports_oct2026](evaluation_reports_oct2026) |

## Month-wise record

The repository is organised by month. Each folder below holds that month's work, a short README (work done, findings, results) and its evaluation reports:
[December 2025](2025-12_December) · [January 2026](2026-01_January) · [February 2026](2026-02_February) · [March 2026](2026-03_March) · [April 2026](2026-04_April) · [May 2026](2026-05_May) · [June 2026](2026-06_June) · [July 2026](2026-07_July) · [August 2026](2026-08_August) · [September 2026](2026-09_September)

## Results

| Crop | Model | Eval set | Accuracy | Macro-F1 | Report |
|---|---|---|---:|---:|---|
| Tomato | Custom CNN 32x32 (TF/Keras) | tomato test, 2,411 (overlaps its training images) | 98.51% | 0.9830 | [report](evaluation_reports_oct2026/keras/custom_cnn_best_tomato_test) |
| Tomato | MobileNetV2 (Keras) | held-out val, 3,198 | 90.90% | 0.8939 | [report](evaluation_reports_oct2026/keras/mobilenetv2_keras) |
| Tomato | EfficientNetB3 (Keras) | held-out val, 3,198 | 93.59% | 0.9275 | [report](evaluation_reports_oct2026/keras/efficientnetb3_keras) |
| Tomato | ConvNeXt-Tiny (Keras) | held-out val, 3,198 | 96.15% | 0.9507 | [report](evaluation_reports_oct2026/keras/convnext_tiny_keras) |
| Tomato | ConvNeXt-Tiny (PyTorch) | test, 2,411 | 99.92% | 0.9991 | [report](evaluation_reports_oct2026/convnext_torch) |
| Tomato | LLaVA-1.6-Mistral-7B LoRA | PlantVillage val, 200 | 2.00% | 0.0151 | [report](evaluation_reports_oct2026/llava_v16) |
| Tomato | BLIP-2 OPT-2.7B LoRA (Mar) | PlantVillage val, 200 | 89.00% | 0.8928 | [report](evaluation_reports_oct2026/blip2_march) |
| Tomato | Florence-2-base-ft zero-shot | PlantVillage val, 200 | 0.00% | 0.0000 | [report](evaluation_reports_oct2026/florence_zeroshot) |
| Tomato | Qwen2.5-VL-7B zero-shot | PlantVillage val, 200 | 8.50% | 0.0800 | [report](evaluation_reports_oct2026/qwen_zeroshot) |
| Tomato | BLIP-2 OPT-2.7B LoRA, 1-epoch re-run (Jun) | PlantVillage val, 200 | 89.00% | 0.8948 | [report](evaluation_reports_oct2026/blip2_june) |
| Tomato | Qwen2.5-VL-7B LoRA r32 (Apr) | PlantVillage val, 200 | 98.50% | 0.9827 | [report](2026-04_April/qwen/evaluation_report.json) |
| Tomato | Qwen2.5-VL-7B LoRA, PlantDx 15-class (May) | PlantVillage val, 200 | 98.00% | 0.9777 | [report](2026-05_May/PlantDx/outputs/evaluations/report.json) |
| Tomato | Qwen2.5-VL-7B-4bit QLoRA, knowledge-grounded (MLX) | test, 910 | 93.70% | 0.919 | [results](https://github.com/iAakash1/experimentation/tree/refactor/professional-research-layout/research/mlx_pipeline) |
| Tomato | PlantVLM, 150 expert captions (Aug) | val, 6 images | 33.3% retrieval | 0.278 | [report](2026-08_August/newpipe/runs/run_002_fold0/evaluation/val_report.json) |
| Tomato (TIH) | Qwen2.5-VL-7B QLoRA 'strong' | test, 649 | 87.98% | 0.8991 | [report](2026-09_September/plantdx_and_tih_tomato/reports/tih_tomato_qwen_qlora_strong/test_finetuned/metrics.json) |
| Tomato (TIH) | Qwen2.5-VL-7B QLoRA 'moreaccurate' | test, 649 | 92.76% | 0.9378 | [report](2026-09_September/plantdx_and_tih_tomato/reports/tih_tomato_moreaccurate/test_finetuned/metrics.json) |
| Tomato (TIH) | 'moreaccurate', external | Dataset/testing, 176 | 69.32% | 0.7195 | [report](2026-09_September/plantdx_and_tih_tomato/reports/tih_tomato_moreaccurate_external/metrics.json) |
| Mango | Qwen2.5-VL-7B-4bit QLoRA (MLX) | test, 200 | 82.00% | 0.811 | [results](https://github.com/iAakash1/experimentation/tree/refactor/professional-research-layout/research/mlx_pipeline) |
| Soybean | ResNet50 (PyTorch) | test, 560 | 91.61% | 0.7814 (Healthy-class F1) | [metrics](2026-01_January/soybean_soynet/results/experiments/resnet50) |
| Soybean | EfficientNetB3 (PyTorch) | test, 560 | 93.93% | 0.8317 (Healthy-class F1) | [metrics](2026-01_January/soybean_soynet/results/experiments/efficientnet_b3) |
| Soybean | EfficientNetB0 (PyTorch) | test, 560 | 93.39% | 0.8195 (Healthy-class F1) | [metrics](2026-01_January/soybean_soynet/results/experiments/efficientnet_b0) |
| Soybean | MobileNetV2 (PyTorch) | test, 560 | 92.50% | 0.8000 (Healthy-class F1) | [metrics](2026-01_January/soybean_soynet/results/experiments/mobilenetv2) |
| Soybean | ResNet50 (Keras) | test, 558 | 95.16% | 0.8556 (Healthy-class F1) | [report](2026-01_January/soybean_soynet/results/experiments/final_report.md) |

## Publication

**Efficient QLoRA Fine-Tuning of Qwen2.5-VL for Multi-Class Tomato Disease Classification and Cross-Domain Evaluation**. Submission 953, accepted on 1 October 2026 (registration and fees completed on 21 October 2026) for the Proceedings of ICAIN 2026 (Springer LNNS series; indexed in
zbMATH, Scopus, DBLP), among the top 20% of reviewed papers. Based on the TIH tomato work in
[`2026-09_September/plantdx_and_tih_tomato`](2026-09_September/plantdx_and_tih_tomato).

## Repository layout

| Folder | Contents |
|---|---|
| `2025-12_December/` | TIH foundation tasks; custom CNN and Keras pipeline (`keras_pipeline_kamaaksha`) |
| `2026-01_January/` | `cottoncrop_repo_snapshot` (preprocessing, crop validation, Keras transfer learning), `pytorch_convnext` (with split file), `soybean_soynet` |
| `2026-02_February/` | PoCRA video editing and the AI for Agri conference (README) |
| `2026-03_March/` | `outputs` (LLaVA-1.6 and BLIP-2 LoRA adapters) |
| `2026-04_April/` | `qwen` (Qwen2.5-VL LoRA, 98.50%), `florence`, `other`, `latest_changes` (production pipeline) |
| `2026-05_May/` | `PlantDx` (15 PlantVillage classes, 98.00%) |
| `2026-06_June/` | `tp` (BLIP-2 1-epoch re-run) |
| `2026-07_July/` | `plantdx_mlx_results` (knowledge-grounded MLX results: tomato 93.7%, mango 82.0%) |
| `2026-08_August/` | `newpipe` (PlantVLM on expert captions), `siglip_openclip_tih_prototype` |
| `2026-09_September/` | `plantdx_and_tih_tomato` (PlantDx package and TIH tomato QLoRA: code, configs, adapters, reports) |
| `evaluation_reports_oct2026/` | Re-evaluation of every saved model (reports, predictions, confusion matrices, scripts) |
| `docs/` | PoCRA reports, PlantDx history, presentations, VLM comparison, timeline verification |

Original development history lives in the source repositories:
[iAakash1/experimentation](https://github.com/iAakash1/experimentation),
[iAakash1/CottonCrop](https://github.com/iAakash1/CottonCrop),
[iAakash1/VLM](https://github.com/iAakash1/VLM),
[kharulkamaaksha/Crop_Disease_VLM](https://github.com/kharulkamaaksha/Crop_Disease_VLM),
[kharulkamaaksha/VLM](https://github.com/kharulkamaaksha/VLM),
[kharulkamaaksha/Tomato_12th_Aug_2026](https://github.com/kharulkamaaksha/Tomato_12th_Aug_2026).

## Research papers studied (March to April 2026)

1. K. Nguyen Quoc, L. Le Thi Thu, L.-D. Quach, *A Vision-Language Foundation Model for Leaf Disease Identification*, 2025. [arXiv:2505.07019](https://arxiv.org/abs/2505.07019)
2. M. Gupta, P. Desai, R. Greer, *Agro-Consensus: Semantic Self-Consistency in Vision-Language Models for Crop Disease Management in Developing Countries*, 2025. [arXiv:2510.21757](https://arxiv.org/abs/2510.21757)
3. M. Z. Hossain et al., *A Lightweight and Explainable Vision-Language Framework for Crop Disease Visual Question Answering*, 2026. [arXiv:2601.05143](https://arxiv.org/abs/2601.05143)
4. Z. Zhao et al., *Image Fusion via Vision-Language Model (FILM)*, ICML 2024. [arXiv:2402.02235](https://arxiv.org/abs/2402.02235)
5. H. Liu et al., *Visual Instruction Tuning (LLaVA)*, NeurIPS 2023. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
6. J. Li et al., *BLIP-2*, ICML 2023. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
7. S. Bai et al., *Qwen2.5-VL Technical Report*, 2025. [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
8. P. Panchal et al., *Deep Learning-Driven Image Captioning: Progress Through Transformers and Large Language Models*, PLOS ONE, 2026. [doi:10.1371/journal.pone.0345012](https://doi.org/10.1371/journal.pone.0345012)

## Notes

- Code is kept exactly as it was run; scripts contain the original absolute paths of the training machines.
  Point them at your own dataset copy before running.
- The knowledge-grounded MLX experiments (tomato and mango, July 2026) ran on an Apple Silicon Mac. Their
  results, confusion matrices and configs are in [iAakash1/experimentation → research/mlx_pipeline](https://github.com/iAakash1/experimentation/tree/refactor/professional-research-layout/research/mlx_pipeline).
- Intermediate training checkpoints (`checkpoint-*`, `epoch-*`) are omitted; the selected best adapters are included.

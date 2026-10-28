# Evaluation reports (October 2026)

Every saved model from the fellowship was re-evaluated on the GPU workstation (NVIDIA RTX 4500 Ada) with the
split and preprocessing it was trained with, so each experiment has a complete, auditable report.

| Folder | Model | Evaluation set | Script |
|---|---|---|---|
| `keras/custom_cnn_*` | Custom CNN 32x32 (Dec 2025) | tomato test split (2,411) and all 15 PlantVillage classes | `keras_eval.py` |
| `keras/mobilenetv2_keras`, `efficientnetb3_keras`, `convnext_tiny_keras` | Keras transfer learning (Jan 2026) | Keras `validation_split=0.2` subset rebuilt exactly (3,198) | `keras_eval.py` |
| `convnext_torch` | ConvNeXt-Tiny PyTorch (Jan 2026) | 70/15/15 split, val 2,397 / test 2,411 | `convnext_torch_eval.py` |
| `llava_v16` | LLaVA-1.6-Mistral-7B LoRA (Mar 2026) | `qwen/captions.json` val[:200] | `vlm_eval.py llava` |
| `blip2_march` | BLIP-2 OPT-2.7B LoRA (Mar 2026) | same 200 | `vlm_eval.py blip2` |
| `blip2_june` | BLIP-2 OPT-2.7B LoRA, 1-epoch re-run (Jun 2026) | same 200 | `vlm_eval.py blip2` |
| `florence_zeroshot` | Florence-2-base-ft, zero-shot (fine-tuning did not complete) | same 200, transformers 4.46.3 | `run_florence_transformers446.py` |
| `qwen_zeroshot` | Qwen2.5-VL-7B-Instruct, zero-shot | same 200 | `vlm_eval.py qwen` |
| `mlx_from_confusion_matrices` | MLX QLoRA tomato and mango (Jul 2026) | counts from the published confusion matrices | - |

The VLM protocol is identical to `2026-04_April/qwen/04_evaluate.py` (April 2026), so all VLM
rows are directly comparable with the 98.50% Qwen2.5-VL LoRA result. Each VLM folder also has `predictions.json`
with the generated text for every image. LLaVA was prompted with each record's own training instruction; the other VLMs used the April evaluation instruction.

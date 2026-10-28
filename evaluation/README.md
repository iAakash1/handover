# Evaluation reports (October 2026)

Every saved model was re-evaluated on the GPU workstation (NVIDIA RTX 4500 Ada) with the split and
preprocessing it was trained with, so each experiment has a complete, auditable report.

| Folder | Model | Evaluation set | How to re-create it |
|---|---|---|---|
| `keras/custom_cnn_*` | Custom CNN 32×32 (Dec 2025) | tomato test split (2,411) and all 15 PlantVillage classes | `projects/01_cnn_keras_tomato/evaluate.py` |
| `keras/mobilenetv2_keras`, `efficientnetb3_keras`, `convnext_tiny_keras` | Keras transfer learning (Jan 2026) | Keras `validation_split=0.2` subset rebuilt exactly (3,198) | `projects/01_cnn_keras_tomato/evaluate.py` |
| `convnext_torch` | ConvNeXt-Tiny PyTorch (Jan 2026) | 70/15/15 split, val 2,397 / test 2,411 | `projects/02_cnn_pytorch_convnext/evaluate.py` |
| `llava_v16` | LLaVA-1.6-Mistral-7B LoRA (Mar 2026) | the 200 validation records of `qwen/captions.json` | `vlm_eval.py llava` |
| `blip2_march` | BLIP-2 OPT-2.7B LoRA (Mar 2026) | same 200 | `vlm_eval.py blip2` |
| `blip2_june` | BLIP-2 OPT-2.7B LoRA, 1-epoch re-run (Jun 2026) | same 200 | `vlm_eval.py blip2` |
| `florence_zeroshot` | Florence-2-base-ft, zero-shot | same 200 (florence env, transformers 4.46.3) | `vlm_eval.py florence` |
| `qwen_zeroshot` | Qwen2.5-VL-7B-Instruct, zero-shot | same 200 | `vlm_eval.py qwen` |
| `mlx_from_confusion_matrices` | MLX QLoRA tomato and mango (Jul 2026) | counts from the published confusion matrices | – |

The VLM protocol is the same as `projects/06_qwen25vl_plantvillage/qwen/04_evaluate.py` (April 2026), so all VLM
rows compare directly with the 98.50% Qwen2.5-VL LoRA result. Each VLM folder has `predictions.json` with the
generated text for every image. LLaVA was prompted with each record's own training instruction; the other VLMs
used the April evaluation instruction.

## Commands (from this folder)

```powershell
$env:MAX_EVAL = 5      # optional: quick check on 5 images instead of 200
..\.venvs\vlm\Scripts\python vlm_eval.py blip2 out_blip2  ..\projects\04_llava_blip2\outputs\blip2_plantvillage\best
..\.venvs\vlm\Scripts\python vlm_eval.py blip2 out_blip2j ..\projects\04_llava_blip2\blip2_rerun_june\outputs\blip2_plantvillage\best
..\.venvs\vlm\Scripts\python vlm_eval.py llava out_llava  ..\projects\04_llava_blip2\outputs\llava_plantvillage\best
..\.venvs\vlm\Scripts\python vlm_eval.py qwen  out_qwen    # base model, no adapter
..\.venvs\vlm\Scripts\python vlm_eval.py qwen  out_qwenft ..\projects\06_qwen25vl_plantvillage\qwen\outputs\qwen25vl_plantvillage\best
..\.venvs\florence\Scripts\python vlm_eval.py florence out_florence
```

`vlm_predict.py` runs the same models on a single photo (see its header for the model names).

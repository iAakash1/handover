# Training plan — qwen25vl_tomato_qlora

> Pre-flight only. **No training has been run.** Launch it yourself with the
> command at the bottom, on the Apple Silicon machine with mlx-vlm installed.

## Model & method
- Model: `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` (4-bit MLX)
- Adapter: **qlora** (rank 16, alpha 32, dropout 0.05, train_vision=False)
- Objective: SFT, train_on_completions=True
- Seed: 20260711 | grad_checkpoint: True | image_resize: 448

## Dataset (tomato only)
- Images discovered: **18006**
- Training rows: **18006** (train 16201 / val 895 / test 910)
- Response pool: frozen corpus `sha256:1225eb8605edb71839f89618bfdcd4a1abba414869f64c1fdc5951ee865ba089`
- Instructions: 8 | captions/image: 1

| disease | rows |
|---|---|
| tomato_bacterial_spot | 2127 |
| tomato_early_blight | 1000 |
| tomato_healthy | 1590 |
| tomato_late_blight | 1756 |
| tomato_leaf_mold | 952 |
| tomato_mosaic_virus | 373 |
| tomato_septoria_leaf_spot | 1771 |
| tomato_spider_mites | 1676 |
| tomato_target_spot | 1404 |
| tomato_yellow_leaf_curl_virus | 5357 |

## Schedule & budget (estimates)
- Effective batch size: **8** (micro 1 x grad-accum 8)
- Iterations: **16201** over 1 epoch(s) (~2025 optimizer updates)
- Estimated wall-clock: **405-945 min** (6.8-15.8 h)
- Estimated peak memory: **8-14 GB** of 24 GB
- Checkpoints: every 200 steps (~81), keep_last=3, keep_best=True
- Estimated disk: **~103 MB** (adapters + dataset JSONL)

### LR schedule preview
| step | lr |
|---|---|
| 0 | 1.00e-06 |
| 3240 | 9.10e-05 |
| 6480 | 6.64e-05 |
| 9720 | 3.56e-05 |
| 12960 | 1.06e-05 |
| 16200 | 1.00e-06 |

## Outputs
- Checkpoints: `checkpoints/qwen25vl_tomato_qlora/`
- Logs (CSV/JSON/Markdown): `logs/qwen25vl_tomato_qlora/`
- This report: `reports/qwen25vl_tomato_qlora/`

## Note on evaluation
mlx-vlm 0.6.x trains on the `train` split; the `validation`/`test` splits are
written and reserved for a separate evaluation milestone (no in-loop eval set).

## The one command to run
```bash
/Users/aakashjawle/miniforge3/bin/python -m mlx_vlm.lora --model-path mlx-community/Qwen2.5-VL-7B-Instruct-4bit --dataset artifacts/training/qwen25vl_tomato_qlora/dataset --split train --train-mode sft --learning-rate 0.0001 --batch-size 1 --epochs 1 --max-seq-length 2048 --gradient-accumulation-steps 8 --grad-clip 1 --steps-per-report 10 --steps-per-eval 200 --steps-per-save 200 --val-batches 8 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 --assistant-id 77091 --output-path checkpoints/qwen25vl_tomato_qlora/adapters.safetensors --image-resize-shape 448 448 --grad-checkpoint --train-on-completions
```

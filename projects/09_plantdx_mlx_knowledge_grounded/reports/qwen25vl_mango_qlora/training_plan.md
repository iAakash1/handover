# Training plan — qwen25vl_mango_qlora

> Pre-flight only. **No training has been run.** Launch it yourself with the
> command at the bottom, on the Apple Silicon machine with mlx-vlm installed.

## Model & method
- Model: `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` (4-bit MLX)
- Adapter: **qlora** (rank 16, alpha 32, dropout 0.05, train_vision=False)
- Objective: SFT, train_on_completions=True
- Seed: 20260711 | grad_checkpoint: True | image_resize: 448

## Dataset (mango)
- Images discovered: **4000**
- Training rows: **4000** (train 3600 / val 200 / test 200)
- Response pool: frozen corpus `sha256:1225eb8605edb71839f89618bfdcd4a1abba414869f64c1fdc5951ee865ba089`
- Instructions: 8 | captions/image: 1

| disease | rows |
|---|---|
| mango_anthracnose | 500 |
| mango_bacterial_canker | 500 |
| mango_cutting_weevil | 500 |
| mango_die_back | 500 |
| mango_gall_midge | 500 |
| mango_healthy | 500 |
| mango_powdery_mildew | 500 |
| mango_sooty_mould | 500 |

## Schedule & budget (estimates)
- Effective batch size: **8** (micro 1 x grad-accum 8)
- Iterations: **3600** over 1 epoch(s) (~450 optimizer updates)
- Estimated wall-clock: **90-210 min** (1.5-3.5 h)
- Estimated peak memory: **8-14 GB** of 24 GB
- Checkpoints: every 200 steps (~18), keep_last=3, keep_best=True
- Estimated disk: **~98 MB** (adapters + dataset JSONL)

### LR schedule preview
| step | lr |
|---|---|
| 0 | 1.00e-06 |
| 720 | 9.25e-05 |
| 1440 | 6.83e-05 |
| 2159 | 3.69e-05 |
| 2879 | 1.10e-05 |
| 3599 | 1.00e-06 |

## Outputs
- Checkpoints: `checkpoints/qwen25vl_mango_qlora/`
- Logs (CSV/JSON/Markdown): `logs/qwen25vl_mango_qlora/`
- This report: `reports/qwen25vl_mango_qlora/`

## Note on evaluation
mlx-vlm 0.6.x trains on the `train` split; the `validation`/`test` splits are
written and reserved for a separate evaluation milestone (no in-loop eval set).

## The one command to run
```bash
/Users/aakashjawle/miniforge3/envs/vlm/bin/python -m mlx_vlm.lora --model-path mlx-community/Qwen2.5-VL-7B-Instruct-4bit --dataset artifacts/training/qwen25vl_mango_qlora/dataset --split train --train-mode sft --learning-rate 0.0001 --batch-size 1 --epochs 1 --max-seq-length 2048 --gradient-accumulation-steps 8 --grad-clip 1 --steps-per-report 10 --steps-per-eval 200 --steps-per-save 200 --val-batches 8 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 --assistant-id 77091 --output-path checkpoints/qwen25vl_mango_qlora/adapters.safetensors --image-resize-shape 448 448 --grad-checkpoint --train-on-completions
```

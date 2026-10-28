# Training checkpoints kept on the TIH workstation

The final (best) model of every run is in this repository's download manifest, and the per-epoch checkpoints of
the TIH tomato runs and PlantVLM come with `python setup/download_models.py --all`. The step-by-step training
checkpoints below add up to about 60 GB, more than GitHub's Git LFS can hold, so they stay on the TIH workstation
(NVIDIA RTX 4500 Ada) at the paths shown. Each is a LoRA adapter saved during training; `best/` of each run is
the one we evaluated and published.

## LLaVA-1.6 LoRA, March 2026 (project 04 / failed_experiments/02)

`C:\Users\Admin\Desktop\VLM\outputs\llava_plantvillage` · 12.8 GB

| Folder | Size | Files |
|---|---:|---|
| `checkpoint-100` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-200` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-300` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-400` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-500` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-600` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-700` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-800` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-900` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1000` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1100` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1200` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1300` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1400` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1500` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1600` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1700` | 709 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `best` | 712 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |

## BLIP-2 LoRA, March 2026 (project 04)

`C:\Users\Admin\Desktop\VLM\outputs\blip2_plantvillage` · 7.3 GB

| Folder | Size | Files |
|---|---:|---|
| `checkpoint-200` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-400` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-600` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-800` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1000` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1200` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1400` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1600` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1800` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-2000` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-2200` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-2400` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-2600` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-2800` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-3000` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-3200` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-3400` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-3600` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-3800` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-4000` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-4200` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-4400` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-4600` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-4800` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-5000` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-5200` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-5400` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-5600` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-5800` | 243 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `best` | 247 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json |

## BLIP-2 LoRA, June re-run (project 04/blip2_rerun_june)

`C:\Users\Admin\Desktop\VLM\tp\outputs\blip2_plantvillage` · 25.2 GB

| Folder | Size | Files |
|---|---:|---|
| `checkpoint-100` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-200` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-300` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-400` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-500` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-600` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-700` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-800` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-900` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-1000` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-1100` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-1200` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-1300` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-1400` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-1500` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-1600` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-1700` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-1800` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-1900` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-2000` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-2100` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-2200` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-2300` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-2400` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-2500` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-2600` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-2700` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-2800` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-2900` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-3000` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-3100` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-3200` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-3300` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `checkpoint-3400` | 734 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json … |
| `best` | 247 MB | README.md, adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json |

## Qwen2.5-VL LoRA, April 2026 (project 06)

`C:\Users\Admin\Desktop\VLM\qwen\outputs\qwen25vl_plantvillage` · 4.2 GB

| Folder | Size | Files |
|---|---:|---|
| `checkpoint-300` | 415 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-600` | 415 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-900` | 415 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1200` | 415 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1500` | 415 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-1800` | 415 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-2100` | 415 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-2400` | 415 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `checkpoint-2700` | 415 MB | README.md, adapter_config.json, adapter_model.safetensors |
| `best` | 426 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |

## PlantDx Qwen2.5-VL, May 2026 (project 07)

`C:\Users\Admin\Desktop\VLM\PlantDx\outputs\checkpoints` · 15.0 GB

| Folder | Size | Files |
|---|---:|---|
| `checkpoint-200` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `checkpoint-400` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `checkpoint-600` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `checkpoint-800` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `checkpoint-1000` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `checkpoint-1200` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `checkpoint-1400` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `checkpoint-1600` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `checkpoint-1800` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `checkpoint-2000` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `checkpoint-2200` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `checkpoint-2400` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |
| `best` | 1,154 MB | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, processor_config.json, tokenizer.json … |

## PlantDx run folders, May 2026 (project 07)

`C:\Users\Admin\Desktop\VLM\PlantDx\outputs\runs` · 0.0 GB

| Folder | Size | Files |
|---|---:|---|
| `run_001` | 0 MB | config.yaml, logs, plots |
| `run_002` | 0 MB | config.yaml, logs, plots |

## PlantVLM second fold, Aug 2026 (project 08)

`C:\Users\Admin\Desktop\VLM\newpipe\runs\run_002_fold0_iso2` · 0.0 GB

| Folder | Size | Files |
|---|---:|---|
| `predictions` | 0 MB | val_base.json |

Also: `C:\Users\Admin\Desktop\VLM\PlantDx\outputs\checkpoints\best\training_state.pt` (762 MB), the optimizer state for resuming the PlantDx run.

Total: 64.4 GB.

# Training Pipeline (Milestone 7) — QLoRA on Qwen2.5-VL (MLX)

A config-driven, deterministic workflow that fine-tunes **Qwen2.5-VL-7B-Instruct-4bit**
on tomato or mango leaf captions using **mlx-vlm** on Apple Silicon (M4 Pro, 24 GB). It
builds the image-grounded training set by cross-joining the **frozen** caption corpus
(the response pool) with the normalized images for that crop, then orchestrates mlx-vlm's
LoRA trainer. It **never** modifies the frozen pipeline (ontology, vocabulary,
concepts, templates, generator, validator, corpus, exporters) and reads image *paths*
only — never pixels, never an LLM/VLM in the data path.

> **Scope:** tomato (10 classes) or mango (8 classes), selected by the run config's
> `data.crop`/`data.classes` (`configs/train/qwen25vl_{tomato,mango}.yaml`). One model:
> Qwen2.5-VL-7B-Instruct-4bit (already downloaded). LoRA/QLoRA are config-driven;
> DoRA is accepted in config for forward-compat but the installed backend
> (mlx-vlm 0.6.x) cannot run it — `train` fails closed with a clear message.

## Prerequisites (run once, in order)

These are the **frozen** upstream stages; the training pipeline consumes their outputs.

```bash
# 1) Normalize the raw images into datasets/<crop>/processed/<class>/
plantdx normalize --dataset tomato
plantdx normalize --dataset mango

# 2) Build the frozen caption corpus (the response pool, covers every crop) -> artifacts/corpus/
plantdx generate
```

Install `plantdx` into the **same** Python environment that has `mlx-vlm`
(Apple Silicon). Training is launched as a subprocess of that interpreter:

```bash
pip install -e .        # in the env where `python -c "import mlx_vlm"` works
```

## Configuration (three composable layers)

| File | Purpose |
|---|---|
| `configs/train/qwen25vl_tomato.yaml` | the tomato run: seed, optimizer, data, checkpoints, logging |
| `configs/train/qwen25vl_mango.yaml` | the mango run — identical LoRA/optimizer/schedule, only crop-specific values (`data.crop`, `data.classes`, `data.image_glob`, `data.instructions_path`, `run_name`) differ |
| `configs/models/qwen25vl.yaml` | the model: repo id, 4-bit, seq length, assistant id, image resize |
| `configs/lora/{qlora,lora,dora}.yaml` | the adapter method + rank/alpha/dropout |

The run YAML references a model and a LoRA file by name (`model: qwen25vl`,
`lora: qlora`); an inline `model_override:` / `lora_override:` mapping overrides
individual keys. Everything is validated fail-closed (ratios sum to 1, qlora needs
a 4-bit base, positive hyperparameters, known backend).

## Commands

```bash
# Build the dataset + plan + report. NEVER launches training.
plantdx prepare-training --config configs/train/qwen25vl_tomato.yaml

# Preview the exact command + plan without launching.
plantdx train --config configs/train/qwen25vl_tomato.yaml --dry-run

# Launch training (this is the real run — hours on M4 Pro).
plantdx train --config configs/train/qwen25vl_tomato.yaml

# Same commands work unchanged for mango — only the config differs.
plantdx train --config configs/train/qwen25vl_mango.yaml --crop mango --dry-run
plantdx train --config configs/train/qwen25vl_mango.yaml --crop mango

# Inference with the trained adapter.
plantdx infer --adapter checkpoints/qwen25vl_tomato_qlora --image leaf.JPG
plantdx infer --adapter checkpoints/qwen25vl_tomato_qlora --folder some_dir/ --json
```

`prepare-training` and `train --dry-run` are side-effect-light (build dataset +
write the report); only `train` **without** `--dry-run` starts training.

## What the data builder produces

`artifacts/training/<run_name>/dataset/` with `train.jsonl`, `validation.jsonl`,
`test.jsonl`, and `manifest.json`. Each JSONL row is exactly what `mlx_vlm.lora`
consumes:

```json
{"image": "/abs/path/leaf.JPG", "question": "<instruction>", "answer": "<caption>"}
```

- **Response** is a caption taken **verbatim** from the frozen corpus for that
  image's disease — nothing is generated here.
- **Instruction** is drawn from `assets/training/instructions.json` by default (task
  prompts only; no disease knowledge, no `<image>` marker — mlx-vlm inserts the image
  token); the mango config points `data.instructions_path` at
  `assets/training/instructions_mango.json` (same prompts, "mango leaf" wording)
  instead of the tomato-worded default.
- **Labels** come from `assets/metadata/label_map.json` (`<class folder> -> disease_id`,
  one section per crop — `tomato` and `mango`).
- **Splits** are image-grouped (no image leaks across splits) and disease-stratified;
  assignment is a pure SHA-256 function of `(split_seed, image_id)`.

Determinism: same config + corpus + image set ⇒ byte-identical JSONL and manifest.

## Determinism & honesty guarantees

- No pixels are read and no LLM/VLM runs while building the dataset — only paths + labels.
- Captions are used verbatim from the frozen corpus; the corpus `content_hash` is
  pinned into `manifest.json` and the report.
- The frozen upstream artifacts are untouched (their golden hashes are unchanged).

## Note on evaluation

mlx-vlm 0.6.x trains on the `train` split and passes `val_dataset=None` in its CLI
path, so `--steps-per-eval` / `--val-batches` do not evaluate a held-out set during
training. The `validation.jsonl` / `test.jsonl` splits are still written and are
reserved for a separate evaluation milestone.

## Outputs

| Directory | Contents |
|---|---|
| `artifacts/training/<run>/dataset/` | `train/validation/test.jsonl` + `manifest.json` |
| `checkpoints/<run>/` | `adapters.safetensors` (final) + periodic snapshots |
| `logs/<run>/` | `metrics.csv/json/md`, `train.log` |
| `reports/<run>/` | `training_plan.md` + `training_plan.json` (with the exact command) |

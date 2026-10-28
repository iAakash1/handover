# 09 · PlantDx knowledge-grounded VLM on Apple Silicon (MLX) — July 2026

> **Runs on a Mac only.** Trained with Apple's MLX framework on an Apple M4 Pro (24 GB unified memory). This folder
> is everything done on that Mac: the full project (code, tests, knowledge base, configs, caption corpus, training
> datasets, evaluation reports, training logs, Streamlit demo), the tomato and mango images it was trained on, and
> the earlier caption and zero-shot experiments (`earlier_experiments/`). Only the two trained adapter files are
> missing; they were no longer on the Mac (see [Adapters](#adapters)).

## What MLX is, in short

MLX is Apple's array framework for Apple Silicon (M1/M2/M3/M4). CPU and GPU share one pool of "unified memory",
so a 4-bit 7B model plus LoRA training fits in 24 GB without an NVIDIA card. `mlx-vlm` adds vision-language
models on top of MLX (loading, LoRA training, generation). MLX only runs on macOS with Apple Silicon — it does
not run on Windows or with CUDA, so nothing in this folder runs on the TIH workstation.

## What we built

Instead of letting a model write training captions (which copies its mistakes), we compiled the captions from a
**cited Disease Knowledge Base**: DKB → typed ontology → controlled vocabulary → per-disease concept models →
templates → 1,966 validated captions. Every step is deterministic and content-hashed, so the caption corpus can
be rebuilt byte for byte. Qwen2.5-VL-7B-Instruct-4bit was then fine-tuned with QLoRA on image + caption pairs.

| Setting | Value |
|---|---|
| Base model | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` |
| Method | QLoRA (LoRA over the 4-bit base), rank 16, alpha 32, dropout 0.05, language model only |
| Training | 1 epoch, LR 1e-4 cosine, 100 warm-up steps, micro-batch 1 × grad-accum 8, seed 20260711 |
| Trainable parameters | 40,370,176 |
| Tomato data | PlantVillage tomato, 10 classes, 16,201 / 895 / 910 (image-grouped split) |
| Mango data | MangoLeafBD, 8 classes, 3,600 / 200 / 200 |
| Hardware | Apple M4 Pro, 24 GB unified memory, Python 3.12.13, `mlx` 0.31.2, `mlx-vlm` 0.6.4 |

## Results (held-out test sets)

| Crop | Base (zero-shot) | Fine-tuned | Macro-F1 | Weighted-F1 | Macro precision / recall* |
|---|---:|---:|---:|---:|---|
| Tomato (910 images) | 7.3% | **93.7%** | 0.919 | 0.937 | 0.937 / 0.912 |
| Mango (200 images) | 17.5% | **82.0%** | 0.811 | 0.811 | 0.920 / 0.820 |

\* computed from the published confusion matrices (`../../evaluation/mlx_from_confusion_matrices`).

Caption quality, tomato (base → fine-tuned): BLEU-4 0.003 → 0.192, ROUGE-L 0.093 → 0.428, CIDEr 0.000 → 0.956,
BERTScore-F1 0.843 → 0.907. Mango: BLEU-4 0.004 → 0.190, ROUGE-L 0.098 → 0.423, METEOR 0.160 → 0.454,
CIDEr 0.000 → 0.946, BERTScore-F1 0.843 → 0.907.

Weakest classes: tomato Early Blight (F1 0.79, confused with Late Blight / Leaf Mold); mango Die Back
(F1 0.08 — 18 of its 25 test images produced no recognisable disease name).

Every number comes from `reports/qwen25vl_<crop>_qlora/evaluation/metrics.json`.

## Timeline of the Mac work

| Date | What | Where |
|---|---|---|
| 12 Jun 2026 | PlantCaption: knowledge-base + VLM caption pipeline | `earlier_experiments/01_plantcaption_pipeline_2026-06-12/` |
| 23 Jun 2026 | Caption sets for the benchmark leaves (four versions) | `earlier_experiments/02_caption_sets_2026-06-23/` |
| 24–25 Jun 2026 | Zero-shot benchmark, 4 VLMs on MLX, 16 images | `earlier_experiments/03_zero_shot_4vlm_benchmark_2026-06-25/` |
| 1 Jul 2026 | Zero-shot benchmark, 7 VLMs on MLX, 30 images | `earlier_experiments/04_zero_shot_7vlm_benchmark_2026-07-01/` |
| 11–13 Jul 2026 | Knowledge base → ontology → vocabulary → concepts → templates → 1,966 captions; dataset audit (12 Jul) and normalisation (13 Jul) | `knowledge_base/`, `src/plantdx/`, `artifacts/`, `datasets/`, `reports/` |
| 13–14 Jul 2026 | QLoRA training on MLX: tomato (finished 14 Jul 07:18), then mango (14 Jul afternoon) | `configs/train/`, `logs/`, `checkpoints/` |
| 14 Jul 2026 | Evaluation, base vs fine-tuned, both crops | `reports/qwen25vl_<crop>_qlora/evaluation/` |
| 16–17 Jul 2026 | Streamlit demo, tried on uploaded photos | `app/`, `streamlit_app.py`, `uploads/`, `predictions/` |

`earlier_experiments/README.md` has the results and run commands of the four earlier experiments.

## Files in this folder

| Path | What it is |
|---|---|
| `src/plantdx/` | The Python package: audit, normalization, ontology, vocabulary, concepts, templates, generation, corpus, dataset, training (`training/mlx_runner.py`), evaluation |
| `app/`, `streamlit_app.py` | Streamlit demo over the trained adapters (upload a leaf → diagnosis, confidence, DKB symptoms) |
| `knowledge_base/` | Stage 1: the cited Disease Knowledge Base (`dkb.json`, `DKB_report.md`) |
| `caption_framework/`, `ontology_design/` | Stage 2 and 3 design specifications |
| `configs/` | Pipeline configs; `configs/train/qwen25vl_tomato.yaml` and `qwen25vl_mango.yaml` are the two training runs |
| `assets/` | Authored inputs: caption templates, label map, instruction banks (tomato, mango) |
| `tests/` | 587 unit / integration / benchmark tests |
| `docs/` | Architecture, training, evaluation, demo app, known issues, ADRs, figures |
| `scripts/` | `setup_eval_env.sh` (evaluation env), `dev_setup.sh`, `run_pipeline.sh` (early scaffold), README figure renderer |
| `artifacts/` | Generated: ontology, vocabulary, concept models, templates, the 1,966-caption corpus, exports (generic / llava / paligemma / blip2 / messages), training datasets (`artifacts/training/qwen25vl_<crop>_qlora/dataset/`) |
| `tomato/raw/PlantVillage/` | The PlantVillage tomato images used: 10 classes, `train/` + `val/`, 18,160 images (311 MB) |
| `mango/raw/MangoLeafBD/` | MangoLeafBD: 8 classes, 4,000 images (142 MB) |
| `datasets/` | Normalization report, class mappings, per-image manifests and dataset cards; `plantdx normalize` writes the normalized image copies to `datasets/<crop>/processed/` (not in git) |
| `reports/` | Dataset audit (class distribution, duplicates, corrupt images, image statistics) and the full evaluation of both runs: metrics, per-disease scores, confusion matrices, BLEU / ROUGE / METEOR / CIDEr / BERTScore, hallucination checks, latency, figures, raw predictions |
| `logs/` | MLX training logs of both runs (`logs/qwen25vl_<crop>_qlora/train.log`) and the demo app log |
| `checkpoints/` | `adapter_config.json` of both runs (the weights are not in git, see below) |
| `predictions/`, `uploads/` | Predictions made with the demo app on 16 Jul 2026 and the photos uploaded to it |
| `results/` | Summary tables and graphs used in the fellowship report |
| `earlier_experiments/` | PlantCaption pipeline, June caption sets, the 4-model and 7-model MLX zero-shot benchmarks |
| `README_original.md` | The README of the original repository (github.com/iAakash1/experimentation) |
| `dot_gitignore.txt` | The original repository's `.gitignore` (renamed so it does not hide the results here) |

On the Mac the project also sat in a nested `experimentation/` folder: an identical copy that was pushed to a
teammate's GitHub repository which no longer exists. Its one commit is kept in
`../../history/kharulkamaaksha_experimentation.bundle`.

## How to start (Mac with Apple Silicon)

Everything below runs from this folder. The original environment was a conda env with Python 3.12.

```bash
cd projects/09_plantdx_mlx_knowledge_grounded
conda create -n vlm python=3.12 -y && conda activate vlm
pip install -e ".[dev,train]"            # plantdx + mlx + mlx-vlm (use mlx-vlm 0.6.4 or newer)
pip install -r app/requirements.txt      # Streamlit, for the demo
pytest                                   # the test suite (on a Mac with a GPU)
```

The first training or inference run downloads the base model `mlx-community/Qwen2.5-VL-7B-Instruct-4bit`
(about 6 GB) from Hugging Face.

### 1. Data

The images are already in place (paths set in `configs/paths.yaml`):

- `tomato/raw/PlantVillage/{train,val}/Tomato___*` — the 10 PlantVillage tomato classes
  (https://github.com/spMohanty/PlantVillage-Dataset). On the Mac this folder also held the other 28 PlantVillage
  classes (apple, corn, grape, …), which the pipeline never used; they are left out here.
- `mango/raw/MangoLeafBD/<class>/` — the 8 MangoLeafBD classes (https://data.mendeley.com/datasets/hxsnvwty3r/1,
  CC BY 4.0).

`plantdx normalize` on these files reproduces the July dataset exactly: tomato 18,160 images, checksum
`b9556420…` (same as `datasets/tomato/dataset_card.md`); mango 4,000 images. The mango card in `datasets/` is from
an earlier run before the mango images were added, so it still shows 0 images.

The training datasets in `artifacts/training/` list each image by its absolute path on the Mac
(`/Users/aakashjawle/Desktop/experiments/datasets/...`). On another Mac, run `plantdx normalize` and then
`plantdx prepare-training --config configs/train/qwen25vl_<crop>.yaml` (`plantdx train` also does this first)
instead of using those files directly.

### 2. Build the caption dataset (deterministic, CPU only)

The outputs of these steps are already in `artifacts/` and `datasets/`; re-running them reproduces the same
content hashes.

```bash
plantdx audit             # inventory the raw datasets -> reports/
plantdx normalize         # canonical datasets/<crop>/processed/
plantdx ontology          # knowledge base -> typed graph
plantdx vocabulary        # controlled vocabulary + symptom lexicon
plantdx concepts          # per-disease concept models
plantdx generate          # the validated caption corpus (1,966 captions)
plantdx corpus --all      # exports: generic / llava / paligemma / blip2 / messages
```

### 3. Train (QLoRA, MLX)

```bash
plantdx train --config configs/train/qwen25vl_tomato.yaml --dry-run    # shows the plan
plantdx train --config configs/train/qwen25vl_tomato.yaml              # writes checkpoints/qwen25vl_tomato_qlora
plantdx train --config configs/train/qwen25vl_mango.yaml --crop mango  # writes checkpoints/qwen25vl_mango_qlora
```

Times estimated from the speeds in `logs/`: tomato 16,201 iterations, about 12 hours (median 0.36 it/s, peak
memory 7.2 GB); mango 3,600 iterations, about 1.6 hours.

### 4. Predict and evaluate

```bash
plantdx infer --adapter checkpoints/qwen25vl_tomato_qlora --image leaf.JPG
streamlit run streamlit_app.py           # the demo; pick tomato or mango in the sidebar
```

Evaluation runs in two stages because the metrics stack and MLX are kept in separate environments:

```bash
# stage 1, in the MLX env: base and fine-tuned predictions on the test split
plantdx evaluate --stage inference --adapter checkpoints/qwen25vl_tomato_qlora \
                 --dataset artifacts/training/qwen25vl_tomato_qlora/dataset \
                 --output-dir reports/qwen25vl_tomato_qlora/evaluation
# stage 2, in a second env: python -m venv .venv-eval && source .venv-eval/bin/activate && ./scripts/setup_eval_env.sh
plantdx evaluate --stage analyze --output-dir reports/qwen25vl_tomato_qlora/evaluation
```

More detail: `docs/TRAINING.md`, `docs/EVALUATION.md`, `docs/DEMO_APP.md`, `docs/KNOWN_ISSUES.md`.

## Adapters

Both runs finished and saved `checkpoints/qwen25vl_<crop>_qlora/adapters.safetensors` (see the last lines of
`logs/qwen25vl_<crop>_qlora/train.log`), but the weight files are no longer on the Mac; only `adapter_config.json`
is left. Their SHA-256 checksums are recorded in `reports/qwen25vl_<crop>_qlora/evaluation/system_info.json`
(tomato `9d4b21f4…`, mango `c944488f…`). To use the demo or `plantdx infer`, retrain with step 3 (same config,
same seed) and put the new `adapters.safetensors` next to `adapter_config.json`. One test needs the original
tomato weights and fails until they are back (`tests/unit/evaluation/test_eval_checkpoint.py::test_real_frozen_tomato_checkpoint_resolves`).

## Code history

The full commit history of this project (tags `v0.1-scaffold` … `v1.0.0`) is in `../../history/experimentation.bundle`;
this folder matches commit `9f08d99` (17 Jul 2026). The later TIH tomato work grew out of the same package and is
in `../10_tih_tomato_qwen_qlora`.

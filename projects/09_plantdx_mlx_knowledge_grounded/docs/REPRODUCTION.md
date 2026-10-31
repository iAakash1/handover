# Reproducing the MLX models from a fresh clone

Run everything from `projects/09_plantdx_mlx_knowledge_grounded/`. Per-model details are in
[`../models/`](../models/README.md); what is in git, downloaded, generated or missing is in [`ARTIFACTS.md`](ARTIFACTS.md).

## 0. Check the clone (any OS)

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/verify_project.py                           # files, configs, CLI, checkpoints, datasets, links
pytest                                                     # 584 tests; metrics-stack tests skip unless [eval] is installed
```

## 1. Environment for training and inference (macOS, Apple Silicon)

```bash
conda create -n vlm python=3.12 -y && conda activate vlm
pip install -e ".[train]" -r requirements-mlx.txt          # mlx 0.31.2, mlx-vlm 0.6.4 as on the Mac
pip install -r app/requirements.txt                        # Streamlit, for the demo
```

A second environment for the metrics stage (never install it next to MLX, see `EVALUATION.md`):

```bash
python3 -m venv .venv-eval && source .venv-eval/bin/activate && ./scripts/setup_eval_env.sh
```

## 2. Data

```bash
plantdx normalize                                                 # raw images -> datasets/<crop>/processed
plantdx prepare-training --config configs/train/qwen25vl_tomato.yaml
plantdx prepare-training --config configs/train/qwen25vl_mango.yaml
```

The caption corpus and every stage before it are already in `artifacts/`; `plantdx ontology`, `vocabulary`,
`concepts`, `generate` rebuild them with the same content hashes.

## 3. Train, infer, evaluate

```bash
plantdx train --config configs/train/qwen25vl_tomato.yaml                  # ~12 h on an M4 Pro, peak 7.2 GB
plantdx train --config configs/train/qwen25vl_mango.yaml --crop mango      # ~1.6 h
plantdx infer --adapter checkpoints/qwen25vl_tomato_qlora --image leaf.JPG
plantdx evaluate --stage inference --adapter checkpoints/qwen25vl_tomato_qlora \
                 --dataset artifacts/training/qwen25vl_tomato_qlora/dataset \
                 --output-dir reports/qwen25vl_tomato_qlora/evaluation     # MLX env
plantdx evaluate --stage analyze --output-dir reports/qwen25vl_tomato_qlora/evaluation   # eval env
```

Use a new `--output-dir` to keep the July evaluation next to a new one.

## What was verified on 31 Oct 2026 (Windows 10, Python 3.10, CPU only, no MLX)

| Step | Result |
|---|---|
| `pip install -e ".[dev,generate]"`, `plantdx --help` and all 15 subcommands' `--help` | work |
| Test suite (584 tests) | 524 passed; 59 skipped (the `[eval]` metrics stack and Streamlit were not installed, plus placeholder tests of unfinished milestones); 1 failed: `test_real_frozen_tomato_checkpoint_resolves`, which needs the missing tomato adapter weights |
| `plantdx normalize` on the images in git | tomato 18,160 images, checksum `b9556420…` (the same as the July dataset card); mango 4,000, checksum `6fa29822…` |
| `plantdx prepare-training` for both configs, compared with the July files in `artifacts/training/` | **every row identical** (same images in the same order, same questions and answers) for train, validation and test of both crops; only the absolute image-path prefix differs |
| `scripts/verify_project.py` | no failures; the two adapter weight files are reported MISSING |

Not verified here, because they need Apple Silicon: training, inference, evaluation inference, the demo's model
backend, and the zero-shot benchmarks.

### One behaviour fix made for this

`plantdx prepare-training` first gave 18,159 tomato images on Windows instead of the July 18,006: the config selects
`*.JPG`, and Windows globbing is case-insensitive, so 153 `.jpg` files and nothing else were added (macOS and Linux
glob case-sensitively). `src/plantdx/training/data/discovery.py` now matches file names case-sensitively on every OS
(`fnmatch.fnmatchcase`), which is exactly what macOS did; results on macOS and Linux do not change. Covered by
`tests/unit/training/test_training_data.py::test_discovery_glob_is_case_sensitive_on_every_os`.

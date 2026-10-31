# Models in this project

Every model trained or tried on the Mac with Apple's MLX framework. Each card lists the model's files, its
environment, and the exact commands for data preparation, training, validation, inference and evaluation.

| Model | What it is | Data (test set) | Result | Weights |
|---|---|---|---|---|
| [`qwen25vl_tomato_qlora`](qwen25vl_tomato_qlora/README.md) | Qwen2.5-VL-7B-Instruct 4-bit + QLoRA (r16), knowledge-grounded captions | PlantVillage tomato, 10 classes (910) | **93.7%**, macro-F1 0.919 (base 7.3%) | missing, retrain |
| [`qwen25vl_mango_qlora`](qwen25vl_mango_qlora/README.md) | same model and method | MangoLeafBD, 8 classes (200) | **82.0%**, macro-F1 0.811 (base 17.5%) | missing, retrain |
| [`zero_shot_benchmarks`](zero_shot_benchmarks/README.md) | 7 off-the-shelf VLMs, no training (4-VLM and 7-VLM runs) | 16 and 30 leaf images | best 23% (InternVL3-8B) | Hugging Face downloads |
| [PlantCaption](../earlier_experiments/01_plantcaption_pipeline_2026-06-12/captioning/README.md) | caption-generation pipeline (knowledge base + Qwen2.5-VL scene description); no training | 20 sample images | sample captions in `sample_outputs/` | Hugging Face download |

## Where a model's files live

The two fine-tuned models share all code (`src/plantdx/`) and differ only in their config. The package reads and
writes run-named folders, so a model's files sit in the same place under each top-level folder:

| | tomato | mango |
|---|---|---|
| Config | `configs/train/qwen25vl_tomato.yaml` | `configs/train/qwen25vl_mango.yaml` |
| Images | `tomato/raw/PlantVillage/` | `mango/raw/MangoLeafBD/` |
| Training data (generated, July copy in git) | `artifacts/training/qwen25vl_tomato_qlora/dataset/` | `artifacts/training/qwen25vl_mango_qlora/dataset/` |
| Checkpoint | `checkpoints/qwen25vl_tomato_qlora/` | `checkpoints/qwen25vl_mango_qlora/` |
| Training log | `logs/qwen25vl_tomato_qlora/train.log` | `logs/qwen25vl_mango_qlora/train.log` |
| Plan and evaluation | `reports/qwen25vl_tomato_qlora/` | `reports/qwen25vl_mango_qlora/` |

These folders were deliberately not moved under `models/`: the CLI defaults, the evaluation defaults, the demo app
and the tests all expect them there (`configs/paths.yaml`, `src/plantdx/evaluation/config.py`). Shared stages
(knowledge base, ontology, vocabulary, caption corpus) are documented in `../docs/` and `../README.md`.

## Same model family elsewhere in the handover

The later TIH tomato work (`../../10_tih_tomato_qwen_qlora`) grew out of this package and still contains its MLX
runner (`src/plantdx/training/mlx_runner.py`), but its own models were trained on CUDA. Its code is newer; the code
that produced the two models above is the one in this folder (commit `9f08d99` of 17 Jul 2026, one documentation
commit after tag `v1.0.0`; both are in `../../../history/experimentation.bundle`).

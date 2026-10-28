# 09 · PlantDx knowledge-grounded VLM on Apple Silicon (MLX) — July 2026

> **These models cannot run on the TIH workstation.** They were trained with Apple's MLX framework on an
> Apple M4 Pro (24 GB unified memory), and the adapters stayed on that Mac. Everything else is here:
> configs, results, figures and the exact commands to reproduce on a Mac.

## What MLX is, in short

MLX is Apple's array framework for Apple Silicon (M1/M2/M3/M4). CPU and GPU share one pool of "unified memory",
so a 4-bit 7B model plus LoRA training fits in 24 GB without an NVIDIA card. `mlx-vlm` adds vision-language
models on top of MLX (loading, LoRA training, generation). MLX only runs on macOS with Apple Silicon — it does
not run on Windows or with CUDA, which is why this section is documentation and results only.

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
| Hardware | Apple M4 Pro, 24 GB unified memory, `mlx-vlm` 0.6.4 |

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

## Files in this folder

| Path | What it is |
|---|---|
| `results/mlx_pipeline/results/benchmark_summary.csv` | The numbers above |
| `results/mlx_pipeline/graphs/` | Confusion matrices, per-disease F1, base vs fine-tuned charts |
| `results/zero_shot/results/baseline_summary.csv` | Zero-shot accuracy of the base model |
| `configs/train/qwen25vl_tomato.yaml`, `configs/train/qwen25vl_mango.yaml` | The two training runs |
| `configs/models/qwen25vl.yaml`, `configs/lora/qlora.yaml` | Model and LoRA settings |

The full source (knowledge base, ontology, caption compiler, MLX training and evaluation) is the `plantdx`
package in `../10_tih_tomato_qwen_qlora/src/plantdx` and, in its exact July state, tag **v1.0.0** of
<https://github.com/iAakash1/experimentation>.

## How to reproduce on a Mac (Apple Silicon, Python 3.10+)

```bash
git clone https://github.com/iAakash1/experimentation && cd experimentation
git checkout v1.0.0                       # July state, includes the mango config
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"                 # mlx + mlx-vlm (use mlx-vlm 0.6.4 or newer for training)
# put PlantVillage under tomato/raw/PlantVillage and MangoLeafBD under mango/raw/MangoLeafBD
plantdx audit                             # inventory the raw datasets
plantdx normalize                         # canonical datasets/<crop>/processed/
plantdx ontology                          # knowledge base -> typed graph
plantdx vocabulary                        # controlled vocabulary + symptom lexicon
plantdx concepts                          # per-disease concept models
plantdx generate                          # the validated caption corpus
plantdx train --config configs/train/qwen25vl_tomato.yaml --dry-run    # shows the plan
plantdx train --config configs/train/qwen25vl_tomato.yaml              # writes checkpoints/qwen25vl_tomato_qlora
plantdx train --config configs/train/qwen25vl_mango.yaml --crop mango
plantdx infer --adapter checkpoints/qwen25vl_tomato_qlora --image leaf.JPG
```

Evaluation runs in two stages (inference needs MLX, metrics need a separate env with `pip install -e ".[eval]"`):

```bash
plantdx evaluate --stage all --adapter checkpoints/qwen25vl_tomato_qlora \
                 --dataset artifacts/training/qwen25vl_tomato_qlora/dataset
```

The Streamlit demo for the adapters is `streamlit run streamlit_app.py` in the same repository.

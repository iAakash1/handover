# Model card: `qwen25vl_mango_qlora`, PlantDx mango leaf captioning (MLX)

Every file this model needs, and the exact command for each stage. Run everything from
`projects/09_plantdx_mlx_knowledge_grounded/`. Same code and method as the tomato model
([`../qwen25vl_tomato_qlora/README.md`](../qwen25vl_tomato_qlora/README.md)); only the data-related values of the
config differ.

## 1. Model and purpose

Describes a mango leaf photo in one diagnostic caption that names the disease and its signs (8 MangoLeafBD
classes: healthy, anthracnose, bacterial canker, cutting weevil, die back, gall midge, powdery mildew, sooty mould).
Trained 14 Jul 2026 on an Apple M4 Pro.

## 2. Architecture and source

Base [`mlx-community/Qwen2.5-VL-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-4bit);
QLoRA with `mlx_vlm.lora`, rank 16, alpha 32 (scale 2.0), dropout 0.05 on q/k/v/o and gate/up/down of all 28
language-model layers (196 modules, `checkpoints/qwen25vl_mango_qlora/adapter_config.json`); vision tower frozen;
loss on answers only.

## 3. Dataset

MangoLeafBD (Ahmed et al., 2023, CC BY 4.0, <https://data.mendeley.com/datasets/hxsnvwty3r/1>), in the repository
at `mango/raw/MangoLeafBD/<Class>/` (4,000 images, 500 per class). All 4,000 match `*.jpg`. Split by image, seed
20260711: **3,600 train / 200 validation / 200 test** (25 test images per class). Questions from
`assets/training/instructions_mango.json`, answers from the same 1,966-caption corpus as tomato.

## 4. Dataset preparation

```bash
plantdx normalize                                                  # -> datasets/mango/processed (not in git)
plantdx prepare-training --config configs/train/qwen25vl_mango.yaml
```

Regenerating reproduces the committed July splits row for row; only the absolute image-path prefix changes
(verified 31 Oct 2026, `../../docs/REPRODUCTION.md`). The mango dataset card in `datasets/mango/` and
`datasets/normalization_report.json` are from a normalization run before the mango images were added (the report
covers tomato only and the card shows 0 images). The July training plan
(`reports/qwen25vl_mango_qlora/training_plan.md`) records 4,000 images, and normalizing today gives 4,000 (dataset
checksum `6fa29822…`, 31 Oct 2026).

## 5. Required files and artifacts

| File | In git? | Role |
|---|---|---|
| `configs/train/qwen25vl_mango.yaml` (+ `configs/models/qwen25vl.yaml`, `configs/lora/qlora.yaml`) | yes | the run definition |
| `mango/raw/MangoLeafBD/` | yes (142 MB) | images |
| `artifacts/corpus/captions.jsonl`, `assets/training/instructions_mango.json`, `assets/metadata/label_map.json` | yes | answers, questions, labels |
| `artifacts/training/qwen25vl_mango_qlora/dataset/*.jsonl` | yes (July originals, Mac paths) | regenerate with step 4 |
| `datasets/mango/processed/` | no, generated | normalized images |
| `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` | no, Hugging Face download | base model |
| `checkpoints/qwen25vl_mango_qlora/adapter_config.json` | yes | adapter definition |
| `checkpoints/qwen25vl_mango_qlora/adapters.safetensors` | **missing** (see section 10) | trained weights |

## 6. Environment

Same as tomato: macOS on Apple Silicon, Python 3.12, `pip install -e ".[train]" -r requirements-mlx.txt`; metrics in a
separate `[eval]` environment (`scripts/setup_eval_env.sh`).

## 7. Configuration (`configs/train/qwen25vl_mango.yaml`)

Identical to tomato except `run_name: qwen25vl_mango_qlora`, `crop: mango`, the 8 classes,
`image_glob: "*.jpg"` and `instructions_path: assets/training/instructions_mango.json`. LR 1e-4 (constant in the
log), micro-batch 1 × gradient accumulation 8, 1 epoch, seed 20260711, max sequence 2,048, images 448 × 448.

## 8. Training

```bash
plantdx train --config configs/train/qwen25vl_mango.yaml --crop mango --dry-run
plantdx train --config configs/train/qwen25vl_mango.yaml --crop mango
```

July command: `reports/qwen25vl_mango_qlora/training_plan.json`; log: `logs/qwen25vl_mango_qlora/train.log`
(3,600 iterations, peak memory 7.4 GB, about 1.6 hours; final loss 0.204).

## 9. Validation

No held-out loss during training (mlx-vlm 0.6.x passes no validation set). Evaluate the validation split with:

```bash
plantdx evaluate --stage inference --adapter checkpoints/qwen25vl_mango_qlora \
                 --dataset artifacts/training/qwen25vl_mango_qlora/dataset --split validation \
                 --output-dir reports/qwen25vl_mango_qlora/evaluation_validation
```

## 10. Checkpoint

`checkpoints/qwen25vl_mango_qlora/` needs `adapter_config.json` + `adapters.safetensors`. The July weights
(sha256 `c944488f…`, `reports/qwen25vl_mango_qlora/evaluation/system_info.json`) and the 200-step snapshots did
not survive on the Mac. Retrain with section 8; `python scripts/verify_project.py` reports presence and checksum.

## 11. Inference

```bash
plantdx infer --adapter checkpoints/qwen25vl_mango_qlora --image path/to/leaf.jpg \
              --instruction "Describe the visible condition of this mango leaf."
streamlit run streamlit_app.py      # choose mango in the sidebar
```

(`plantdx infer` defaults to a tomato instruction, so pass a mango one.)

## 12. Evaluation

```bash
plantdx evaluate --stage inference --adapter checkpoints/qwen25vl_mango_qlora \
                 --dataset artifacts/training/qwen25vl_mango_qlora/dataset \
                 --output-dir reports/qwen25vl_mango_qlora/evaluation
plantdx evaluate --stage analyze --output-dir reports/qwen25vl_mango_qlora/evaluation      # [eval] env
```

## 13. Input and output

A leaf photo (resized to 448 × 448) and an instruction in; one caption out. Predictions:
`reports/qwen25vl_mango_qlora/evaluation/raw/predictions.jsonl`.

## 14. Results (200 test images, `reports/qwen25vl_mango_qlora/evaluation/metrics.json`)

| | Accuracy | Macro-F1 | Weighted-F1 | Macro P / R | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore-F1 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Base (zero-shot) | 17.5% | 0.128 | 0.128 | 0.161 / 0.175 | 0.004 | 0.098 | 0.160 | 0.000 | 0.843 |
| Fine-tuned | **82.0%** | **0.811** | 0.811 | 0.920 / 0.820 | 0.190 | 0.423 | 0.454 | 0.946 | 0.907 |

Fine-tuned latency: median 1.71 s per image (p95 2.80 s), peak memory 12.1 GB. Weakest class: die back, F1 0.08
(1 of 25 right; 18 answers named no disease).

## 15. Limitations and hardware

Apple Silicon only (tested M4 Pro, 24 GB). Weights missing (retrain to use). Only 500 images per class; die back
is effectively not learned.

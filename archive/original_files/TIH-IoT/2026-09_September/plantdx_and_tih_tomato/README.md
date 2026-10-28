# September 2026: PlantDx package and the TIH tomato benchmark (July MLX research included)

**In short:** this folder is the team's [iAakash1/experimentation](https://github.com/iAakash1/experimentation)
repository at its latest branch, `refactor/professional-research-layout` (commit 66a40bc), which already contains the commits of every other branch. It holds two pieces of work:

1. **July 2026: knowledge-grounded PlantDx (Apple Silicon / MLX).** Training captions are built from a cited
   Disease Knowledge Base instead of being written by an AI, and Qwen2.5-VL-7B is then fine-tuned with QLoRA.
   Tomato: 7.3% → **93.7%**. Mango: 17.5% → **82.0%**. These runs were on a Mac, so their results, configs and
   confusion matrices are in the experimentation repo:
   [research/mlx_pipeline](https://github.com/iAakash1/experimentation/tree/refactor/professional-research-layout/research/mlx_pipeline).
2. **August – September 2026: TIH tomato dataset (NVIDIA GPU).** Qwen2.5-VL-7B fine-tuned with 4-bit QLoRA to
   name one of 11 classes (diseases, pests, healthy leaf/fruit) from the TIH `Tomato_12thAug26` images.

| Run | Date | Test accuracy | Macro-F1 | Model | Report |
|---|---|---:|---:|---|---|
| strong (LoRA r32, 392 px) | 15 Sep 2026 | 87.98% | 0.8991 | `checkpoints/tih_tomato_qwen_qlora_strong` | `reports/tih_tomato_qwen_qlora_strong/test_finetuned/metrics.json` |
| moreaccurate (LoRA r64, 532 px) | 16 Sep 2026 | **92.76%** | 0.9378 | `checkpoints/tih_tomato_moreaccurate` | `reports/tih_tomato_moreaccurate/test_finetuned/metrics.json` |
| moreaccurate, cross-domain (176 images from another source) | 17 Sep 2026 | 69.32% | 0.7195 | same | `reports/tih_tomato_moreaccurate_external/metrics.json` |

This TIH work, including the cross-domain benchmark, is the basis of the paper **Efficient QLoRA Fine-Tuning of Qwen2.5-VL for Multi-Class Tomato Disease Classification and Cross-Domain Evaluation**, accepted at ICAIN 2026 (Springer LNNS) on 1 October 2026; registration and fees were completed on 21 October 2026.

The drop on the outside images shows the model is sensitive to new photo conditions. Late Blight was the
hardest class. `RESULTS.md` has the full per-class tables. The original repository README follows.

---

# PlantDx Experimentation

PlantDx is a research repository for reproducible vision-language model experiments in plant
disease classification. It contains independent exploration work and an applied TIH tomato
benchmark, organized so that results, configurations, and evaluation code can be reviewed without
distributing private datasets or unpublished knowledge assets.

## Research tracks

| Track | Scope | Location |
|---|---|---|
| MLX research | Zero-shot exploration and QLoRA fine-tuning on Apple Silicon | [`research/mlx_pipeline/`](research/mlx_pipeline/) |
| Zero-shot baselines | Unadapted VLM classification baselines | [`research/zero_shot/`](research/zero_shot/) |
| TIH tomato benchmark | Qwen2.5-VL QLoRA training and internal/external evaluation on NVIDIA CUDA | [`experiments/tih_tomato_qwen/`](experiments/tih_tomato_qwen/) |

The MLX track is the general research pipeline. The TIH track applies a separate CUDA training and
evaluation path to the TIH `Tomato_12thAug26` dataset. The two tracks keep separate configurations,
hardware assumptions, results, and experiment histories.

## Results

The complete experiment timeline, benchmark tables, interpretation, and figures are in
[`RESULTS.md`](RESULTS.md).

| Benchmark | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TIH internal test | **92.76%** | **93.78%** | **92.86%** |
| TIH external domain-shift test | **69.32%** | **71.95%** | **69.38%** |

## Repository structure

```text
experiments/
  tih_tomato_qwen/       TIH configurations, aggregate results, figures, evaluation tools
research/
  mlx_pipeline/          independent MLX configurations, results, and figures
  zero_shot/             zero-shot experiment metadata and aggregate baselines
src/plantdx/             reusable training, inference, and evaluation implementation
configs/                 canonical runnable configurations
tests/                   unit and integration tests
docs/                    public engineering documentation
private_archive/         ignored local archive for unpublished research components
```

## Reproduce the public evaluation

Python 3.10 or newer is required. CUDA evaluation additionally requires the NVIDIA/PyTorch stack
listed in the project environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"
```

Run the TIH external evaluator after placing the private dataset and trained adapter at the paths
referenced by the experiment configuration:

```bash
python -m plantdx.cli company evaluate-external \
  --config configs/train/qwen25vl_tih_tomato_moreaccurate.yaml \
  --dataset-root Dataset/testing
```

Regenerate the publication-safe aggregate tables and figures from local run outputs:

```bash
python experiments/tih_tomato_qwen/evaluation/generate_public_artifacts.py
```

Run the TIH public tests:

```bash
pytest tests/unit/company
ruff check src/plantdx/company/external_evaluate.py \
  tests/unit/company/test_external_evaluate.py \
  experiments/tih_tomato_qwen/evaluation/generate_public_artifacts.py
```

Tests that require unpublished knowledge inputs are skipped when those local inputs are absent.

## Public release boundary

The public branch includes reusable model code, experiment configurations, evaluation scripts,
aggregate benchmark results, figures, and tests. Raw datasets, model checkpoints, per-image private
predictions, curated knowledge assets, and unpublished reasoning specifications are excluded. The
machine-readable classification is recorded in
[`public_release_manifest.yaml`](public_release_manifest.yaml).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

The source code is licensed under the [Apache License 2.0](LICENSE). Dataset licenses remain with
their original providers; datasets are not redistributed by this repository.

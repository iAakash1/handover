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

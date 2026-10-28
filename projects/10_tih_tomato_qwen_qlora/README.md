# 10 · TIH tomato: Qwen2.5-VL QLoRA, 11 classes, cross-domain benchmark (Aug–Sep 2026) — ICAIN 2026 paper

Qwen2.5-VL-7B-Instruct adapted with 4-bit NF4 QLoRA to name one of 11 classes (leaf and fruit diseases, pests,
healthy leaf / fruit) in the private TIH `Tomato_12thAug26` images. This is the work in our ICAIN 2026 paper.

**Environment:** `tih` (PyTorch 2.7.1 CUDA 11.8, this package installed editable) · **GPU:** NVIDIA 24 GB

| Run | Date | LoRA / image size | Test accuracy | Macro-F1 |
|---|---|---|---:|---:|
| `tih_tomato_qwen_qlora` (smoke test only) | 11 Sep 2026 | r16 / 448 px | – | – |
| `tih_tomato_qwen_qlora_strong` | 15 Sep 2026 | r32 / 392 px | 87.98% | 0.8991 |
| `tih_tomato_moreaccurate` (paper) | 16 Sep 2026 | r64 / 532 px | **92.76%** | 0.9378 |
| moreaccurate on 176 images from another source | 17 Sep 2026 | – | 69.32% | 0.7195 |

## Handover quick start (run from this folder)

```powershell
cd projects\10_tih_tomato_qwen_qlora
$cfg = "configs/train/qwen25vl_tih_tomato_moreaccurate.yaml"
$py  = "..\..\.venvs\tih\Scripts\python"
& $py -m plantdx.cli company predict --config $cfg --adapter checkpoints/tih_tomato_moreaccurate --image <photo>
& $py -m plantdx.cli company evaluate --config $cfg --adapter checkpoints/tih_tomato_moreaccurate --split test
& $py -m plantdx.cli company evaluate-external --config $cfg --adapter checkpoints/tih_tomato_moreaccurate `
      --dataset-root "C:/Users/Admin/Desktop/TIH_IITB_August_2026/Tomato_12thAug26/Dataset/testing"
& $py -m plantdx.cli company preflight --config $cfg     # checks GPU, dataset, split hashes before training
& $py -m plantdx.cli company train --config $cfg         # retrain (~3 h on the RTX 4500)
```

The private dataset is found automatically on the TIH workstation; elsewhere add
`--dataset-root <folder with the 11 class folders>`. Reports land in `reports/`, adapters in `checkpoints/`.
The July MLX (Apple Silicon) work is the same `plantdx` package — see `../09_plantdx_mlx_knowledge_grounded`.

Good to know:
- `evaluate --split test` always runs all 649 test images (about 4 min); `--limit` is refused on the test split on
  purpose so the reported number can't come from a subset. For a quick check use `--split validation --limit 10`.
- Running `evaluate` again overwrites the reports in `reports/tih_tomato_moreaccurate/`; pass `--run-name <new>` to
  keep them (it needs that run's `artifacts/company/<run>/dataset`, so copy the folder first).
- The PEFT "Unexpected keyword arguments ... LoraConfig" warning is harmless: the adapter was saved with a newer PEFT.

---

*Original project README:*

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

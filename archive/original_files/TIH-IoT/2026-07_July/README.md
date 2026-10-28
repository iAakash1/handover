# July 2026: knowledge-grounded PlantDx (tomato and mango)

## Work done

- 150 expert-revised captions finalised (1 Jul); PoCRA report submitted (10 Jul).
- Knowledge-grounded PlantDx on Apple Silicon/MLX (11–17 Jul): captions compiled from a cited Disease Knowledge Base, then QLoRA fine-tuning of Qwen2.5-VL-7B for tomato and mango.

## Findings

- Tomato: 7.3% → **93.7%** (macro-F1 0.919). Mango: 17.5% → **82.0%** (macro-F1 0.811).
- Mango Die Back is the weak class (recall 0.04): its single-leaf symptoms overlap other diseases.
- Full results and figures: [iAakash1/experimentation → research/mlx_pipeline](https://github.com/iAakash1/experimentation/tree/refactor/professional-research-layout/research/mlx_pipeline).

## Results

| Model / evaluation set | Accuracy | Macro-F1 |
|---|---:|---:|
| Tomato MLX QLoRA (910) | 93.70% | 0.9190 |
| Mango MLX QLoRA (200) | 82.00% | 0.8110 |

## Evaluation reports in this folder

| File | Copied from |
|---|---|
| [`mlx_from_confusion_matrices__evaluation_report.json`](evaluation_reports/mlx_from_confusion_matrices__evaluation_report.json) | `evaluation_reports_oct2026/mlx_from_confusion_matrices/evaluation_report.json` |
| [`mlx_pipeline__results__benchmark_summary.csv`](evaluation_reports/mlx_pipeline__results__benchmark_summary.csv) | `2026-07_July/plantdx_mlx_results/mlx_pipeline/results/benchmark_summary.csv` |
| [`zero_shot__results__baseline_summary.csv`](evaluation_reports/zero_shot__results__baseline_summary.csv) | `2026-07_July/plantdx_mlx_results/zero_shot/results/baseline_summary.csv` |
| [`mlx_pipeline__graphs__mango_confusion_matrix.png`](evaluation_reports/mlx_pipeline__graphs__mango_confusion_matrix.png) | `2026-07_July/plantdx_mlx_results/mlx_pipeline/graphs/mango_confusion_matrix.png` |
| [`mlx_pipeline__graphs__mango_metrics_comparison.png`](evaluation_reports/mlx_pipeline__graphs__mango_metrics_comparison.png) | `2026-07_July/plantdx_mlx_results/mlx_pipeline/graphs/mango_metrics_comparison.png` |
| [`mlx_pipeline__graphs__mango_per_disease_f1.png`](evaluation_reports/mlx_pipeline__graphs__mango_per_disease_f1.png) | `2026-07_July/plantdx_mlx_results/mlx_pipeline/graphs/mango_per_disease_f1.png` |
| [`mlx_pipeline__graphs__tomato_confusion_matrix.png`](evaluation_reports/mlx_pipeline__graphs__tomato_confusion_matrix.png) | `2026-07_July/plantdx_mlx_results/mlx_pipeline/graphs/tomato_confusion_matrix.png` |
| [`mlx_pipeline__graphs__tomato_metrics_comparison.png`](evaluation_reports/mlx_pipeline__graphs__tomato_metrics_comparison.png) | `2026-07_July/plantdx_mlx_results/mlx_pipeline/graphs/tomato_metrics_comparison.png` |
| [`mlx_pipeline__graphs__tomato_per_disease_f1.png`](evaluation_reports/mlx_pipeline__graphs__tomato_per_disease_f1.png) | `2026-07_July/plantdx_mlx_results/mlx_pipeline/graphs/tomato_per_disease_f1.png` |

## Code and models

- [MLX results (experimentation repo)](https://github.com/iAakash1/experimentation/tree/refactor/professional-research-layout/research/mlx_pipeline)

# September 2026: TIH private tomato data and cross-domain benchmark

## Work done

- Qwen2.5-VL-7B fine-tuned with 4-bit QLoRA on the TIH 11-class tomato dataset (5,247 / 649 / 649 split): smoke run 11 Sep, 'strong' 15 Sep, 'moreaccurate' 16 Sep.
- Cross-domain benchmark on 176 images from a different source (17 Sep).
- This work became the ICAIN 2026 paper, accepted on 1 October 2026 (Springer LNNS).

## Findings

- 'moreaccurate' (LoRA r64, 532 px) reached **92.76%** test accuracy (macro-F1 0.9378), +4.8 points over 'strong' (87.98%).
- Cross-domain accuracy dropped to 69.32%: Late Blight fell to 14.29% F1 (confused with Target Leaf Spot); Mealy Bugs and White Fly transferred best.
- Robustness to new photo conditions is the main open problem.

## Results

| Model / evaluation set | Accuracy | Macro-F1 |
|---|---:|---:|
| 'strong' QLoRA, test (649) | 87.98% | 0.8991 |
| 'moreaccurate' QLoRA, test (649) | 92.76% | 0.9378 |
| 'moreaccurate', cross-domain (176) | 69.32% | 0.7195 |

## Evaluation reports in this folder

| File | Copied from |
|---|---|
| [`tih_tomato_qwen_qlora_strong__test_finetuned__metrics.json`](evaluation_reports/tih_tomato_qwen_qlora_strong__test_finetuned__metrics.json) | `2026-09_September/plantdx_and_tih_tomato/reports/tih_tomato_qwen_qlora_strong/test_finetuned/metrics.json` |
| [`tih_tomato_qwen_qlora_strong__test_finetuned__confusion_matrix.json`](evaluation_reports/tih_tomato_qwen_qlora_strong__test_finetuned__confusion_matrix.json) | `2026-09_September/plantdx_and_tih_tomato/reports/tih_tomato_qwen_qlora_strong/test_finetuned/confusion_matrix.json` |
| [`tih_tomato_moreaccurate__test_finetuned__metrics.json`](evaluation_reports/tih_tomato_moreaccurate__test_finetuned__metrics.json) | `2026-09_September/plantdx_and_tih_tomato/reports/tih_tomato_moreaccurate/test_finetuned/metrics.json` |
| [`tih_tomato_moreaccurate__test_finetuned__confusion_matrix.json`](evaluation_reports/tih_tomato_moreaccurate__test_finetuned__confusion_matrix.json) | `2026-09_September/plantdx_and_tih_tomato/reports/tih_tomato_moreaccurate/test_finetuned/confusion_matrix.json` |
| [`reports__tih_tomato_moreaccurate_external__metrics.json`](evaluation_reports/reports__tih_tomato_moreaccurate_external__metrics.json) | `2026-09_September/plantdx_and_tih_tomato/reports/tih_tomato_moreaccurate_external/metrics.json` |
| [`reports__tih_tomato_moreaccurate_external__classification_report.csv`](evaluation_reports/reports__tih_tomato_moreaccurate_external__classification_report.csv) | `2026-09_September/plantdx_and_tih_tomato/reports/tih_tomato_moreaccurate_external/classification_report.csv` |
| [`reports__tih_tomato_moreaccurate_external__confusion_matrix.png`](evaluation_reports/reports__tih_tomato_moreaccurate_external__confusion_matrix.png) | `2026-09_September/plantdx_and_tih_tomato/reports/tih_tomato_moreaccurate_external/confusion_matrix.png` |
| [`tih_tomato_qwen__graphs__tih_qwen_benchmark_overview.png`](evaluation_reports/tih_tomato_qwen__graphs__tih_qwen_benchmark_overview.png) | `2026-09_September/plantdx_and_tih_tomato/experiments/tih_tomato_qwen/graphs/tih_qwen_benchmark_overview.png` |
| [`tih_tomato_qwen__graphs__tih_qwen_external_confusion_matrix.png`](evaluation_reports/tih_tomato_qwen__graphs__tih_qwen_external_confusion_matrix.png) | `2026-09_September/plantdx_and_tih_tomato/experiments/tih_tomato_qwen/graphs/tih_qwen_external_confusion_matrix.png` |
| [`tih_tomato_qwen__graphs__tih_qwen_internal_confusion_matrix.png`](evaluation_reports/tih_tomato_qwen__graphs__tih_qwen_internal_confusion_matrix.png) | `2026-09_September/plantdx_and_tih_tomato/experiments/tih_tomato_qwen/graphs/tih_qwen_internal_confusion_matrix.png` |
| [`tih_tomato_qwen__graphs__tih_qwen_per_class_f1.png`](evaluation_reports/tih_tomato_qwen__graphs__tih_qwen_per_class_f1.png) | `2026-09_September/plantdx_and_tih_tomato/experiments/tih_tomato_qwen/graphs/tih_qwen_per_class_f1.png` |
| [`tih_tomato_qwen__graphs__tih_qwen_training_metrics.png`](evaluation_reports/tih_tomato_qwen__graphs__tih_qwen_training_metrics.png) | `2026-09_September/plantdx_and_tih_tomato/experiments/tih_tomato_qwen/graphs/tih_qwen_training_metrics.png` |

## Code and models

- [TIH adapters](../2026-09_September/plantdx_and_tih_tomato/checkpoints)
- [all TIH reports](../2026-09_September/plantdx_and_tih_tomato/reports)

# January 2026: transfer learning and soybean

## Work done

- MobileNetV2, EfficientNetB3 and ConvNeXt-Tiny transfer learning in Keras (4–6 Jan).
- ConvNeXt-Tiny in PyTorch on the RTX 4500 GPU (8 Jan).
- Crop validation (green-pixel ratio + embedding similarity) and the full validate → preprocess → predict pipeline (21–27 Jan).
- SoyNet soybean healthy-vs-diseased training on the VDI with the provided pipeline, Keras and PyTorch runs (27 Jan).

## Findings

- ConvNeXt-Tiny (PyTorch) was the best CNN: 99.92% test accuracy, macro-F1 0.9991.
- Keras held-out validation: ConvNeXt 96.15%, EfficientNetB3 93.59%, MobileNetV2 90.90%.
- Soybean: Keras ResNet50 95.16%; PyTorch EfficientNetB3 93.93%. The metrics' positive class is Healthy; the PyTorch models find every healthy leaf but over-predict it.
- PlantVillage is near-solved for CNNs on clean lab images, which motivated the move to VLMs and harder datasets.

## Results

| Model / evaluation set | Accuracy | Macro-F1 |
|---|---:|---:|
| ConvNeXt-Tiny PyTorch, test (2,411) | 99.92% | 0.9991 |
| ConvNeXt-Tiny Keras, val (3,198) | 96.15% | 0.9507 |
| EfficientNetB3 Keras, val (3,198) | 93.59% | 0.9275 |
| MobileNetV2 Keras, val (3,198) | 90.90% | 0.8939 |
| SoyNet resnet50 PyTorch, test (560) | 91.61% | 0.7814 |
| SoyNet mobilenetv2 PyTorch, test (560) | 92.50% | 0.8000 |
| SoyNet efficientnet_b0 PyTorch, test (560) | 93.39% | 0.8195 |
| SoyNet efficientnet_b3 PyTorch, test (560) | 93.93% | 0.8317 |

## Evaluation reports in this folder

| File | Copied from |
|---|---|
| [`convnext_torch__README.md`](evaluation_reports/convnext_torch__README.md) | `evaluation_reports_oct2026/convnext_torch/README.md` |
| [`convnext_torch__confusion_matrix_test.png`](evaluation_reports/convnext_torch__confusion_matrix_test.png) | `evaluation_reports_oct2026/convnext_torch/confusion_matrix_test.png` |
| [`convnext_torch__evaluation_report.json`](evaluation_reports/convnext_torch__evaluation_report.json) | `evaluation_reports_oct2026/convnext_torch/evaluation_report.json` |
| [`keras__convnext_tiny_keras__confusion_matrix.png`](evaluation_reports/keras__convnext_tiny_keras__confusion_matrix.png) | `evaluation_reports_oct2026/keras/convnext_tiny_keras/confusion_matrix.png` |
| [`keras__convnext_tiny_keras__evaluation_report.json`](evaluation_reports/keras__convnext_tiny_keras__evaluation_report.json) | `evaluation_reports_oct2026/keras/convnext_tiny_keras/evaluation_report.json` |
| [`keras__efficientnetb3_keras__confusion_matrix.png`](evaluation_reports/keras__efficientnetb3_keras__confusion_matrix.png) | `evaluation_reports_oct2026/keras/efficientnetb3_keras/confusion_matrix.png` |
| [`keras__efficientnetb3_keras__evaluation_report.json`](evaluation_reports/keras__efficientnetb3_keras__evaluation_report.json) | `evaluation_reports_oct2026/keras/efficientnetb3_keras/evaluation_report.json` |
| [`keras__mobilenetv2_keras__confusion_matrix.png`](evaluation_reports/keras__mobilenetv2_keras__confusion_matrix.png) | `evaluation_reports_oct2026/keras/mobilenetv2_keras/confusion_matrix.png` |
| [`keras__mobilenetv2_keras__evaluation_report.json`](evaluation_reports/keras__mobilenetv2_keras__evaluation_report.json) | `evaluation_reports_oct2026/keras/mobilenetv2_keras/evaluation_report.json` |
| [`results__experiments__final_report.md`](evaluation_reports/results__experiments__final_report.md) | `2026-01_January/soybean_soynet/results/experiments/final_report.md` |
| [`results__experiments__model_comparison.json`](evaluation_reports/results__experiments__model_comparison.json) | `2026-01_January/soybean_soynet/results/experiments/model_comparison.json` |
| [`experiments__resnet50__metrics.json`](evaluation_reports/experiments__resnet50__metrics.json) | `2026-01_January/soybean_soynet/results/experiments/resnet50/metrics.json` |
| [`experiments__mobilenetv2__metrics.json`](evaluation_reports/experiments__mobilenetv2__metrics.json) | `2026-01_January/soybean_soynet/results/experiments/mobilenetv2/metrics.json` |
| [`experiments__efficientnet_b0__metrics.json`](evaluation_reports/experiments__efficientnet_b0__metrics.json) | `2026-01_January/soybean_soynet/results/experiments/efficientnet_b0/metrics.json` |
| [`experiments__efficientnet_b3__metrics.json`](evaluation_reports/experiments__efficientnet_b3__metrics.json) | `2026-01_January/soybean_soynet/results/experiments/efficientnet_b3/metrics.json` |
| [`experiments__resnet50__confusion_matrix.json`](evaluation_reports/experiments__resnet50__confusion_matrix.json) | `2026-01_January/soybean_soynet/results/experiments/resnet50/confusion_matrix.json` |
| [`experiments__mobilenetv2__confusion_matrix.json`](evaluation_reports/experiments__mobilenetv2__confusion_matrix.json) | `2026-01_January/soybean_soynet/results/experiments/mobilenetv2/confusion_matrix.json` |
| [`experiments__efficientnet_b0__confusion_matrix.json`](evaluation_reports/experiments__efficientnet_b0__confusion_matrix.json) | `2026-01_January/soybean_soynet/results/experiments/efficientnet_b0/confusion_matrix.json` |
| [`experiments__efficientnet_b3__confusion_matrix.json`](evaluation_reports/experiments__efficientnet_b3__confusion_matrix.json) | `2026-01_January/soybean_soynet/results/experiments/efficientnet_b3/confusion_matrix.json` |

## Code and models

- [CNN baselines](../2026-01_January)
- [SoyNet](../2026-01_January/soybean_soynet)

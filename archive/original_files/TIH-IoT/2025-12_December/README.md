# December 2025: foundation tasks and first CNN

## Work done

- TIH Week 1 (2–5 Dec): image operations with OpenCV/PIL and from scratch in NumPy (resize, crop, rotate, flip, colour spaces, blur, interpolation, Sobel, Canny, sharpening) with speed comparisons.
- TIH Week 2 (8–12 Dec): training preprocessing and augmentation; image-quality checks for prediction (blur, brightness, contrast, noise, resolution, format).
- TIH Week 3 (15–19 Dec): CNN architecture, activations and losses; CNN trained from scratch on PlantVillage (29 Dec).

## Findings

- The 32×32 custom CNN reached 98.51% on the tomato test split (macro-F1 0.9830) and 86.77% over all 15 PlantVillage classes; indicative only, because these images largely overlap its training data (its own Mac split could not be rebuilt).
- The quality checks written this month became the inference gate used in every later pipeline.

## Results

| Model / evaluation set | Accuracy | Macro-F1 |
|---|---:|---:|
| Custom CNN, tomato test (2,411) | 98.51% | 0.9830 |
| Custom CNN, 15 PlantVillage classes (20,638) | 86.77% | 0.8508 |

## Evaluation reports in this folder

| File | Copied from |
|---|---|
| [`keras__custom_cnn_best_tomato_test__confusion_matrix.png`](evaluation_reports/keras__custom_cnn_best_tomato_test__confusion_matrix.png) | `evaluation_reports_oct2026/keras/custom_cnn_best_tomato_test/confusion_matrix.png` |
| [`keras__custom_cnn_best_tomato_test__evaluation_report.json`](evaluation_reports/keras__custom_cnn_best_tomato_test__evaluation_report.json) | `evaluation_reports_oct2026/keras/custom_cnn_best_tomato_test/evaluation_report.json` |
| [`keras__custom_cnn_best_plantvillage15__confusion_matrix.png`](evaluation_reports/keras__custom_cnn_best_plantvillage15__confusion_matrix.png) | `evaluation_reports_oct2026/keras/custom_cnn_best_plantvillage15/confusion_matrix.png` |
| [`keras__custom_cnn_best_plantvillage15__evaluation_report.json`](evaluation_reports/keras__custom_cnn_best_plantvillage15__evaluation_report.json) | `evaluation_reports_oct2026/keras/custom_cnn_best_plantvillage15/evaluation_report.json` |

## Code and models

- [CNN code and model](../2025-12_December/keras_pipeline_kamaaksha)
- [Week-2 preprocessing module](../2026-01_January/cottoncrop_repo_snapshot/preprocessing)

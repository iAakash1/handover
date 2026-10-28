# SoyNet Multi-Model Experiment Report

**Generated:** 2026-01-27 14:30:38

## Executive Summary

This report presents the results of training four deep learning backbone architectures
for soybean disease detection on the SoyNet dataset using GPU-accelerated training.

### Dataset
- **Disease samples:** 3164
- **Healthy samples:** 556
- **Image size:** (256, 256)
- **Class imbalance ratio:** ~5.7:1 (Disease:Healthy)

### Training Configuration
- **Epochs:** 50
- **Batch size:** 32
- **Early stopping patience:** 15
- **Mixed precision:** Enabled

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Size (MB) |
|-------|----------|-----------|--------|----------|---------|-----------|
| resnet50 | 0.9516 | 0.7692 | 0.9639 | 0.8556 | 0.9903 | 244.8 |
| efficientnetb0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| efficientnetb3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| mobilenetv2 | 0.8907 | 0.7292 | 0.4217 | 0.5344 | 0.8748 | 27.0 |

---

## Recommendations

### Best Overall Accuracy
**resnet50**

### Best Recall (Disease Safety Critical)
**resnet50**

> **Note:** In disease detection, high recall is crucial to minimize false negatives 
> (missing diseased plants). A model with slightly lower accuracy but higher recall 
> may be preferred for real-world agricultural applications.

### Best Deployment Candidate
**mobilenetv2**

This recommendation balances accuracy with model size for efficient deployment
on edge devices or mobile applications.

---

## Key Observations

- Dataset is imbalanced (Disease: 3164, Healthy: 556). Recall is critical for disease detection safety.

---

## Next Steps

1. **GUI Development:** Implement user interface for inference
2. **API Integration:** Create REST API for model serving
3. **Mobile Deployment:** Optimize MobileNetV2 for on-device inference
4. **Paper Writing:** Document methodology and results for publication

---

## Artifacts Generated

- Trained models: `results/experiments/{model_name}/`
- Evaluation plots: `results/experiments/{model_name}/plots/`
- Grad-CAM outputs: `results/experiments/{model_name}/xai/`
- Comparison data: `results/experiments/model_comparison.json`

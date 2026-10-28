# January 2026: soybean (SoyNet)

**In short:** we trained four CNNs to tell **healthy** from **diseased** soybean leaves, using the SoyNet dataset
and the training pipeline provided for the task, on the VDI GPU machine (27 January 2026).

## Data
- SoyNet camera photos, resized to 256×256: 3,164 diseased + 556 healthy images (about 6 diseased for every healthy).
- Split 70 / 15 / 15 with seed 42 → 2,603 train, 557 validation, 560 test.
  The exact list is in [`splits/soynet_pytorch_split_70_15_15_seed42.csv`](splits/soynet_pytorch_split_70_15_15_seed42.csv).
- Images are not included in this repository.

## Two runs on the same day
1. **Keras run** (`main.py`, `run_experiments.py`): ResNet50 and MobileNetV2 trained; the EfficientNet models did not build.
2. **PyTorch run** (`run_pytorch.py`, `pytorch/`): all four backbones trained.

## Results

The "positive" class in these metrics is **Healthy**, so precision, recall and F1 describe the healthy class.

| Model | Run | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| ResNet50 | Keras | **95.16%** | 0.7692 | 0.9639 | 0.8556 | 0.9903 |
| MobileNetV2 | Keras | 89.07% | 0.7292 | 0.4217 | 0.5344 | 0.8748 |
| EfficientNetB3 | PyTorch | 93.93% | 0.7119 | 1.0000 | 0.8317 | 0.9875 |
| EfficientNetB0 | PyTorch | 93.39% | 0.6942 | 1.0000 | 0.8195 | 0.9897 |
| MobileNetV2 | PyTorch | 92.50% | 0.6667 | 1.0000 | 0.8000 | 0.9889 |
| ResNet50 | PyTorch | 91.61% | 0.6412 | 1.0000 | 0.7814 | 0.9866 |

## Where things are
- `results/experiments/<model>/`: metrics, confusion matrix, training history, Grad-CAM images and the trained model (`model.pth`).
- `results/experiments/final_report.md`, `model_comparison.json`: Keras run summary.
- `saved_models/`: Keras models.
- `logs/`: every training log.

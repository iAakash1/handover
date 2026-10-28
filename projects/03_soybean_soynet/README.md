# 03 · Soybean healthy vs diseased — SoyNet (27 January 2026)

**Environment:** `soynet` (TensorFlow 2.10 + PyTorch 2.5.1, CUDA 12.1)

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

## Run it (from the repository root)

```powershell
cd projects\03_soybean_soynet
..\..\.venvs\soynet\Scripts\python predict_soynet.py --image "C:\Users\Admin\Desktop\latest\SoyNet\Preprocessing_SoyNet_Data\Camera Clicks_256_256\Healthy_Preprocessing data\soy (1).JPG"
..\..\.venvs\soynet\Scripts\python predict_soynet.py --image <photo> --model resnet50     # or efficientnet_b0 / mobilenetv2
```

## Train again

```powershell
..\..\.venvs\soynet\Scripts\python run_pytorch.py        # PyTorch run: all four backbones -> results\experiments\<model>\
..\..\.venvs\soynet\Scripts\python run_experiments.py    # Keras run (ResNet50, MobileNetV2)
..\..\.venvs\soynet\Scripts\python main.py               # single Keras model, chosen by MODEL_NAME in config.py
```

All settings live in `config.py` (Keras) and `pytorch/config.py` (PyTorch): image size 256, batch 32, up to 50
epochs, Adam 1e-4 then fine-tuning at 1e-5, early stopping (patience 15), mixed precision. The dataset path comes
from `SOYNET_DIR` (default: the original folder on the TIH workstation).

## Where things are
- `results/experiments/<model>/`: metrics, confusion matrix, training history, Grad-CAM images and the trained model (`model.pth`).
- `results/experiments/final_report.md`, `model_comparison.json`: Keras run summary.
- `saved_models/`: Keras models.
- `logs/`: every training log.

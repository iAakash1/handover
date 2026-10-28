# 02 · ConvNeXt-Tiny in PyTorch (8 January 2026) — best CNN

The same tomato task in PyTorch on the RTX 4500 GPU: ConvNeXt-Tiny from ImageNet weights, fully fine-tuned with
AdamW (LR 1e-4), mixed precision, batch 64, 10 epochs (about 0.9 min per epoch).

**Environment:** `vlm` (any PyTorch with torchvision works)

## Results

| Split | Images | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Validation | 2,397 | 99.79% | 0.9979 |
| Test | 2,411 | **99.92%** | 0.9991 |

Only 2 test images are wrong. Report and confusion matrix: `../../evaluation/convnext_torch/`.

## Run it (from the repository root)

```powershell
cd projects\02_cnn_pytorch_convnext
..\..\.venvs\vlm\Scripts\python predict.py --image ..\01_cnn_keras_tomato\sample.jpg
..\..\.venvs\vlm\Scripts\python evaluate.py        # val + test reports into eval_out\ (~1 min on GPU)
```

## Train again

Open `notebooks/01_preprocessing.ipynb` (makes the 70/15/15 split into `TOMATO_SPLIT_DIR`) and then
`notebooks/02_training_convnext.ipynb` with the `vlm` environment as the kernel. The notebooks use `../data` —
change `DATA_DIR` in the second cell to your `TOMATO_SPLIT_DIR`.

## Files

| Path | What it is |
|---|---|
| `models/convnext_tomato_best.pth` | Trained weights (downloaded by `setup/download_models.py`) |
| `splits/split_70_15_15.csv` | Every image in train/val/test, so the split can be rebuilt exactly |
| `splits/tomato_classes.txt` | The 10 classes |
| `notebooks/` | Split + training notebooks with their outputs |
| `outputs/confusion_matrix.png` | Confusion matrix saved by the notebook |

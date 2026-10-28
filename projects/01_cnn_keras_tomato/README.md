# 01 · Tomato leaf CNNs in Keras (December 2025 – January 2026)

My first models: a small CNN written from scratch for the December TIH tasks, then transfer learning with
MobileNetV2, EfficientNetB3 and ConvNeXt-Tiny on the 10 PlantVillage tomato classes. The `preprocessing/` package
is the Week-2 TIH task (training preprocessing, augmentation, image-quality checks), and `validation/` is the
"is this even a crop leaf?" check.

**Environment:** `keras` (TensorFlow 2.20 / Keras 3, runs on CPU)

## Results

| Model | Evaluated on | Accuracy | Macro-F1 |
|---|---|---:|---:|
| ConvNeXt-Tiny | held-out 20% validation, 3,198 images | **96.15%** | 0.9507 |
| EfficientNetB3 | same | 93.59% | 0.9275 |
| MobileNetV2 | same | 90.90% | 0.8939 |
| Custom CNN (32×32) | tomato test split, 2,411 images* | 98.51% | 0.9830 |

\* indicative only — the CNN's own 80/20 split was drawn on my Mac and these test images overlap its training
data. Full reports: `../../evaluation/keras/`.

## Run it (from the repository root, PowerShell)

```powershell
cd projects\01_cnn_keras_tomato
..\..\.venvs\keras\Scripts\python predict.py --image sample.jpg                 # ConvNeXt (best)
..\..\.venvs\keras\Scripts\python predict.py --image sample.jpg --model efficientnet
..\..\.venvs\keras\Scripts\python predict.py --image sample.jpg --model mobilenet
..\..\.venvs\keras\Scripts\python predict.py --image sample.jpg --model cnn
..\..\.venvs\keras\Scripts\python predict.py --image sample.jpg --quality-check   # reject bad photos first
..\..\.venvs\keras\Scripts\python evaluate.py      # re-creates every number above into eval_out\ (~5 min)
```

The original January demos still work too: `main.py` (EfficientNetB3), `main_mobilenet.py`, `main_convnext.py`.
They run the quality gate and then predict `sample.jpg`.

## Train again

Each `main_*.py` trains its model automatically when the model file is missing, so to retrain one, move its
`.keras` file out of `model\` first (training writes a new file with the same name):

```powershell
Move-Item model\best_model_convnext.keras model\best_model_convnext.old.keras
..\..\.venvs\keras\Scripts\python main_convnext.py          # trains on TOMATO_DATASET_DIR, then predicts sample.jpg
..\..\.venvs\keras\Scripts\python model\train_cnn.py        # custom CNN; set CNN_TRAIN_DIR to a folder of class folders
```

The transfer-learning scripts use a 80/20 `validation_split`, 224×224 input, flips/rotation/zoom, a classifier
head for 10 epochs at LR 1e-3, then the top 20–30 backbone layers for 10 epochs at LR 1e-5, with early stopping.
They only read the 10 tomato folders, even if the dataset folder also has pepper and potato.

## Files

| Path | What it is |
|---|---|
| `predict.py`, `evaluate.py` | Clean entry points (same preprocessing as training) |
| `model/train_*.py` | Training scripts (CNN, MobileNetV2, ConvNeXt, `train_transfer.py` = EfficientNetB3) |
| `model/*.keras` | Trained models (downloaded by `setup/download_models.py`) |
| `preprocessing/` | Week-2 module: `preprocess_training`, `augment_training`, `quality_checks`, `preprocess_inference` |
| `validation/`, `reference/`, `generate_reference_embeddings.py` | Crop check: green-pixel ratio + MobileNetV2 embedding similarity |
| `splitDataset.py` | The 80/20 split I used for the custom CNN |
| `presentation.ipynb`, `sample.jpg` | Notebook and demo image |

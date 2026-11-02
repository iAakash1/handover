# Handover guide: tomato leaf CNNs in Keras (project 01)

**1. What it does.** Classifies a tomato leaf photo into one of 10 PlantVillage classes (9 diseases + healthy). It
also contains the December 2025 TIH preprocessing module and a "is this a crop leaf?" check.

**2. Architecture.** Four models, one code base:

| Model | Input | Design |
|---|---|---|
| Custom CNN | 32×32 | three Conv-BatchNorm-MaxPool blocks (32/64/128 filters), Dense 128, dropout 0.5, softmax |
| MobileNetV2, EfficientNetB3, ConvNeXt-Tiny | 224×224 | ImageNet backbone + new classifier head (transfer learning) |

**3. Dataset.** PlantVillage, the 10 tomato folders (`TOMATO_DATASET_DIR`, see [`../../docs/datasets.md`](../../docs/datasets.md)). The
transfer models use Keras `validation_split=0.2` (3,198 validation images); the custom CNN used its own 80/20
split (`splitDataset.py`).

**4. Training.** Transfer models: classifier head 10 epochs at LR 1e-3, then the top 20–30 backbone layers 10 epochs
at LR 1e-5, flips/rotation/zoom, early stopping (`model/train_mobilenet.py`, `train_transfer.py` = EfficientNetB3,
`train_convnext.py`). Custom CNN: Adam 1e-4, batch 32, up to 50 epochs, early stopping (`model/train_cnn.py`).

**5. Weights.** `model/best_model_cnn.keras`, `best_model_mobilenet.keras`, `best_model_efficientNetB3.keras`,
`best_model_convnext.keras` (+ `final_model.keras`) and `reference/reference_crop_embeddings.npy`. Not in git:
`python setup\download_models.py 01_cnn` fetches them (checksummed).

**6. Install.** `python setup\setup_envs.py` builds `.venvs\keras` (TensorFlow 2.20, Keras 3.12; CPU is enough).

**7. Inference.**
```powershell
cd projects\01_cnn_keras_tomato
..\..\.venvs\keras\Scripts\python predict.py --image sample.jpg                       # ConvNeXt (default)
..\..\.venvs\keras\Scripts\python predict.py --image <photo> --model efficientnet --quality-check
```
`--model` is `convnext`, `efficientnet`, `mobilenet` or `cnn`; `--quality-check` rejects blurry, dark or noisy photos
first. Output: the top classes with probabilities.

**8. Reproduce evaluation.** `..\..\.venvs\keras\Scripts\python evaluate.py` rebuilds the validation split and writes
reports to `eval_out\`. Reference results ([`../../evaluation/keras/`](../../evaluation/keras/README.md)): ConvNeXt-Tiny 96.15% (macro-F1 0.9507),
EfficientNetB3 93.59% (0.9275), MobileNetV2 90.90% (0.8939), custom CNN 98.51% on the tomato test split (indicative:
those images overlap its training data).

**9. Modify / extend.** Hyperparameters are at the top of each `model/train_*.py`; retrain by moving a `.keras` file
out of `model\` and running its `main_*.py` (it trains when the file is missing). Quality-check thresholds:
`preprocessing/quality_checks.py`; crop-check threshold: `validation/crop_validation_embedding.py`; class list:
`TOMATO_CLASSES` in `predict.py`. A new class needs its image folder and a retrain.

**10. Limitations.** PlantVillage photos are lab images on plain backgrounds; field accuracy was not measured. The
custom CNN's 98.51% is not a clean held-out score. Earlier versions of this project are in
[`../../archive/previous_versions/01_cnn_keras_tomato/`](../../archive/previous_versions/01_cnn_keras_tomato/README.md).

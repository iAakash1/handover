# Handover guide: SoyNet, soybean healthy vs diseased (project 03)

**1. What it does.** Classifies a soybean leaf photo as healthy or diseased.

**2. Architecture.** Four ImageNet backbones with a binary head: ResNet50, MobileNetV2 (Keras and PyTorch runs),
EfficientNetB0 and EfficientNetB3 (PyTorch run; via `timm` names in `pytorch/models.py`).

**3. Dataset.** SoyNet camera photos preprocessed to 256×256: 3,164 diseased + 556 healthy (`SOYNET_DIR`, see
[`../../docs/datasets.md`](../../docs/datasets.md)). Stratified 70/15/15, seed 42: 2,603 / 557 / 560, exported to
`splits/soynet_pytorch_split_70_15_15_seed42.csv`. The raw images and the dataset itself are not in git.

**4. Training.** Batch 32, up to 50 epochs, Adam 1e-4 then fine-tuning at 1e-5, early stopping (patience 15), mixed
precision, augmentation on training data only. Settings: `config.py` (Keras), `pytorch/config.py` (PyTorch).

**5. Weights.** PyTorch: `results/experiments/<model>/model.pth`. Keras: `saved_models/*.keras` and
`results/experiments/<model>/soynet_<model>_best.keras`. All fetched by `python setup\download_models.py 03_soybean`.

**6. Install.** `.venvs\soynet` (TensorFlow 2.10 and PyTorch 2.5.1 with CUDA 12.1), built by `setup\setup_envs.py`.

**7. Inference.**
```powershell
cd projects\03_soybean_soynet
..\..\.venvs\soynet\Scripts\python predict_soynet.py --image <photo>                     # EfficientNetB3 (default)
..\..\.venvs\soynet\Scripts\python predict_soynet.py --image <photo> --model resnet50    # or efficientnet_b0 / mobilenetv2
```

**8. Reproduce evaluation.** Evaluation runs at the end of training on the test split and writes
`results/experiments/<model>/metrics.json`, the confusion matrix and Grad-CAM images:
`..\..\.venvs\soynet\Scripts\python run_pytorch.py` (all four PyTorch backbones) or `run_experiments.py` (Keras).
Reference results (positive class = Healthy): Keras ResNet50 **95.16%** (ROC-AUC 0.9903), MobileNetV2 89.07%;
PyTorch EfficientNetB3 93.93%, EfficientNetB0 93.39%, MobileNetV2 92.50%, ResNet50 91.61%.

**9. Modify / extend.** Image size, epochs, learning rates and the list of backbones are in the two config files;
`main.py` trains a single Keras model chosen by `MODEL_NAME` in `config.py`. New backbones go into
`ModelFactory.MODEL_MAP` (`pytorch/models.py`).

**10. Limitations.** The classes are imbalanced (about 6:1); every PyTorch model has Healthy recall 1.0 at lower
precision. The Keras EfficientNet models never built (see
[`../../failed_experiments/10_soynet_keras_efficientnet/`](../../failed_experiments/10_soynet_keras_efficientnet/README.md)). A further soybean model
trained on the TIH VDI with TIH data stayed on the VDI and is not in this repository.

# Handover guide: ConvNeXt-Tiny in PyTorch (project 02)

**1. What it does.** Classifies a tomato leaf photo into one of the 10 PlantVillage tomato classes. It is the most
accurate CNN of the project.

**2. Architecture.** torchvision ConvNeXt-Tiny with ImageNet weights, classifier replaced for 10 classes, fully
fine-tuned.

**3. Dataset.** The 10 tomato classes split 70/15/15 per class: 11,203 train / 2,397 validation / 2,411 test. Every
image's split is listed in `splits/split_70_15_15.csv`; the class order in `splits/tomato_classes.txt`. Data folder:
`TOMATO_SPLIT_DIR` ([`../../docs/datasets.md`](../../docs/datasets.md)).

**4. Training.** AdamW, LR 1e-4, mixed precision, batch 64, 10 epochs on an NVIDIA RTX 4500 Ada (about 9 minutes):
`notebooks/02_training_convnext.ipynb` (the split is made by `notebooks/01_preprocessing.ipynb`).

**5. Weights.** `models/convnext_tomato_best.pth` (111 MB), fetched by `python setup\download_models.py 02_cnn`.

**6. Install.** Uses `.venvs\vlm` (PyTorch 2.5.1, CUDA 12.1), built by `python setup\setup_envs.py`. Runs on CPU too,
slowly.

**7. Inference.**
```powershell
cd projects\02_cnn_pytorch_convnext
..\..\.venvs\vlm\Scripts\python predict.py --image ..\01_cnn_keras_tomato\sample.jpg
```

**8. Reproduce evaluation.** `..\..\.venvs\vlm\Scripts\python evaluate.py` evaluates validation and test into
`eval_out\`. Reference ([`../../evaluation/convnext_torch/`](../../evaluation/convnext_torch/README.md)): validation 99.79%, **test 99.92%**,
macro-F1 0.9991 (2 of 2,411 test images wrong).

**9. Modify / extend.** Learning rate, epochs and augmentation are in the training notebook (`DATA_DIR` in its second
cell points at the split). For more classes, add their folders to the split, regenerate the CSV with notebook 01, retrain,
and update `CLASSES` in `predict.py` (the classifier's output size is `len(CLASSES)`, in ImageFolder order).

**10. Limitations.** Near-perfect on PlantVillage, which is a clean lab dataset; performance on field photos was
not measured.

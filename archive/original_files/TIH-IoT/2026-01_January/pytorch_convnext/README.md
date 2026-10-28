# PyTorch ConvNeXt-Tiny (8 Jan 2026)

**What is here:** `notebooks/01_preprocessing.ipynb` makes the 70/15/15 tomato split (10 classes); `02_training_convnext.ipynb` fine-tunes ConvNeXt-Tiny on the RTX 4500 GPU (10 epochs, ~0.9 min each). `splits/split_70_15_15.csv` lists every image in each split; `models/convnext_tomato_best.pth` is the trained model.

**Results:** Validation 99.79%, test **99.92%** (2,411 images), macro-F1 0.9991.

**Finding:** Best CNN result of the fellowship; PlantVillage is close to solved for CNNs.

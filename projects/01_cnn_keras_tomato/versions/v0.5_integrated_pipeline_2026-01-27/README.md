# Keras pipeline: custom CNN and transfer learning (29 Dec 2025 – 27 Jan 2026)

**What is here:** `model/train_cnn.py` is the CNN built from scratch for the December Week-3 task (3 conv blocks, 32×32 input). `model/train_*.py` are the transfer-learning scripts and `model/*.keras` the trained models. `config.py` runs the full pipeline: crop check → preprocessing → prediction (rejects predictions below 60% confidence).

**Results:** Custom CNN 98.51% on the tomato test split (macro-F1 0.9830; indicative, as these images largely overlap its training data). Report: `evaluation_reports_oct2026/keras/`.

**Finding:** Even a small 32×32 CNN fits clean PlantVillage photos well; a true held-out score needs its original split.
